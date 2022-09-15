/* 
   evaluator program

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>
#include <mikenet/simulator.h>
#include "simconfig.h"
#include "model.h"
#include "model.c"


typedef struct
{
  char ch;
  Real vector[PHO_FEATURES];
} Phoneme;

Phoneme phonemes[50];
int phocount=0;

int symbol_hash[255];

void get_name(char *tag, char *name)
{
  char *p;
  p=strstr(tag,"Word:");
  p+= 5;
  p=strtok(p," \t\n");
  strcpy(name,p);
}

void load_phonemes()
{
  FILE * f;
  char line[255],*p;
  int i,x;
  f=fopen("mapping","r");
  if (f==NULL)
    {
      fprintf(stderr,"no mapping file\n");
      exit(1);
    }
  x=0;
  fgets(line,255,f);
  while(!feof(f))
    {
      p=strtok(line," \t\n");
      if (p[0]=='-')
	p[0]='_';
      phonemes[phocount].ch=p[0];
      symbol_hash[(unsigned int)(p[0])]=x++;
      for(i=0;i<PHO_FEATURES;i++)
	{
	  p=strtok(NULL," \t\n");
	  if (strcmp(p,"NaN")==0)
	    phonemes[phocount].vector[i]= -10;
	  else 
	    phonemes[phocount].vector[i]= atof(p);
	}
      phocount++;
      fgets(line,255,f);
    }
  fclose(f);
}

float euclid_distance(Real *x1,Real *x2)
{
  float d=0,r;
  int i;
  for(i=0;i<PHO_FEATURES;i++)
    {
      r = x1[i] - x2[i];
      d += r * r;
    }
  return d;
}
      
void euclid(Real *v,char *out)
{
  int i,j;
  int nearest_item;
  float error=0;
  float nearest_distance=1000000000.0,d;

  for(i=0;i<PHO_SLOTS;i++)
    {

      nearest_item=-1;
      for(j=0;j<phocount;j++)
	{
	  d=euclid_distance(&v[i*PHO_FEATURES],phonemes[j].vector);
	  if ((nearest_item == -1) ||
	      (d < nearest_distance))
	    {
	      nearest_item=j;
	      nearest_distance=d;
	    }
	}
      error += d;
      out[i]=phonemes[nearest_item].ch;
    }
  out[PHO_SLOTS]=0;
}


int main(int argc,char *argv[])
{
  char euclid_output[100],target_output[100];
  Real sse,totalSse;
  FILE *f;
  Real clampNoise=0.1;
  char feature[1000][30];
  char name[255];
  int time=4;
  int wrongs=0,right=0,wrong=0,wrongtot;
  char weightFile[4000];
  char patfile[4000];
  int reset=1;
  int i,j,count;
  Real error;
  Real epsilon,range;
  int runfor=20;
  Example *ex;  
  int semLesion=0;
  int phoLesion=0;
  int PDLesion=0;
  float lesion;
  int seed=-1;
  float target[PHO_FEATURES*PHO_SLOTS+1];


  /* don't buffer output */
  setbuf(stdout,NULL);

  /* what are the command line arguments? */
  for(i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-seed")==0)
	{
	  seed=atoi(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-epsilon",5)==0)
	{
	  epsilon=atof(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-clampNoise")==0)
	{
	  clampNoise=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-noreset",5)==0)
	{
	  reset=0;
	}
      else if (strncmp(argv[i],"-runfor",5)==0)
	{
	  runfor=atoi(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-pl",3)==0)
	{
	  phoLesion=1;
	  lesion=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-pd",3)==0)
	{
	  PDLesion=1;
	  lesion=atof(argv[i+1]);
	  i++;
	}
      else if (strncmp(argv[i],"-sl",3)==0)
	{
	  semLesion=1;
	  lesion=atof(argv[i+1]);
	  i++;
	}
     else if (strncmp(argv[i],"-time",5)==0)
	{
	  time=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-range")==0)
	{
	  range=atof(argv[i+1]);
	  i++;
	}
      else if ((strncmp(argv[i],"-weight",5)==0) ||
	       (strncmp(argv[i],"-load",5)==0))
	{
	  strcpy(weightFile,argv[i+1]);
	  i++;
	}

      else if ((strncmp(argv[i],"-pat",5)==0))
	{
	  strcpy(patfile,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-errorRadius")==0)
	{
	  default_errorRadius=atof(argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown argument: %s\n",argv[i]);
	  exit(-1);
	}
    }
  
  /* build the network */
  build_model();


  default_errorComputation=SUM_SQUARED_ERROR;

  load_weights(net,weightFile);
  hearing_examples=load_examples("hearing.pat",SAMPLES);
  speaking_examples=load_examples(patfile,SAMPLES);

  default_errorRadius = 0.0;

  load_phonemes();
  error=0.0;
  count=1;
  wrongs=0;
  wrongtot=0;
  totalSse=0;

  /* loop for ITER number of times */
  for(i=0;i<speaking_examples->numExamples;i++)
    {
      ex=get_random_example(hearing_examples);
      crbp_forward(net,ex);
      //crbp_compute_gradients(net,ex);
      //crbp_update_taos(net);

      /* get j'th example from exampleset */
      ex=&speaking_examples->examples[i];
      get_name(ex->name,name);
      printf("%s\t",name);

      /* do forward propagation */
      sse=0.0;
      crbp_forward(net,ex);
      //crbp_compute_gradients(net,ex);
      //crbp_update_taos(net);
      //crbp_apply_deltas(net);
      sse=compute_error(net,ex);
      printf("%.4f ",sse);
      wrong=0;
      totalSse+=sse;

      for(j=0;j<phonOut->numUnits;j++)
	target[j]=get_value(ex->targets,phonOut->index,11,j); 
      euclid(target,target_output);

      right=0;
      for (j=2; j < SAMPLES; j++){
	euclid(phonOut->outputs[j],euclid_output);
	if (strcmp(target_output,euclid_output)==0) {
	  printf("%s\t%s\t%d\t%.2f\n",euclid_output,target_output,j,sse);
	  right=1;
	  j=SAMPLES;
	}
      }


      if (right==0) {
	printf("%s\t%s\t%d\t%.2f\t[WRONG]\n",euclid_output,target_output,j,sse);
	wrongs++;
      }
      
      
    }

  printf("%d wrong\t%.2f mean SSE\n",wrongs,(totalSse/(float)i));
  return 0;
}

