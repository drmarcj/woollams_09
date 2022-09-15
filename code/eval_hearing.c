/* 
   evaluator for semantics

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mikenet/simulator.h>
#include "simconfig.h"
#include "model.c"
#include "split.c"


#ifdef unix
#include <unistd.h>
#endif

#include <mikenet/simulator.h>


int semcount=0;

typedef struct
{
  char ch[256];
  Real vector[SEM_FEATURES];
} Semantics;

Semantics semantics[SEM_WORDS*2];



void get_name(char *tag, char *name)
{
  char *p;
  p=strstr(tag,"Word:");
  p+= 5;
  p=strtok(p," \t\n");
  strcpy(name,p);
}

void load_sem()
{
  FILE * f;
  char line[(SEM_FEATURES*2) + 128];
  char *fields[SEM_FEATURES + 3];
  char *p;
  int i,x,nof;

  f=fopen("sem_mapping","r");
  if (f==NULL)
    {
      fprintf(stderr,"no mapping file\n");
      exit(1);
    }
  x=0;
  semcount=0;
  fgets(line,sizeof(line),f);
  while(!feof(f))
    {
      nof = split(line,fields,sizeof(fields)," \n\t");
      if (nof < SEM_FEATURES) 
	Error0("Oops, not enough fields in semmapping file, line %d\n"
		,semcount);

      strcpy(semantics[semcount].ch,fields[1]);

      for(i=0;i<SEM_FEATURES;i++)
	{
	  if (strcmp(fields[i+2],"NaN")==0)
	    semantics[semcount].vector[i]= -10;
	  else 
	    semantics[semcount].vector[i]= atof(fields[i+2]);
	}
      semcount++;
      fgets(line,sizeof(line),f);
    }
  fclose(f);

}

float euclid_distance(Real *x1,Real *x2)
{
  float d=0,r;
  int i;
  for(i=0;i<SEM_FEATURES;i++)
    {
      r = x1[i] - x2[i];
      d += r * r;
    }
  return d;
}
      
void euclid(Real *v,char *out)
{
  int i;
  int nearest_item;
  float error=0;
  float nearest_distance=1000000,d;

  // matches the word that is closest in Euclidean space to the actual output

  for(i=0;i<semcount;i++) {
    d=euclid_distance(&v[0],semantics[i].vector);

    if (d < nearest_distance)	{
      nearest_item=i;
      nearest_distance=d;
    }
  }

  strcpy(out,semantics[nearest_item].ch); 

  return;
}


int main(int argc,char *argv[])
{
  char target_output[256],euclid_output[256];
  Real target[SEM_FEATURES];
  Real sse;
  FILE *f;
  Real clampNoise=0.1;
  char feature[1000][30];
  char name[255];
  int time=4;
  int wrongs=0,wrong=0,wrongtot;
  char weightFile[4000];
  char patfile[4000];
  int reset=1;
  ExampleSet *examples;
  Example *ex;
  int i,j,count;
  Real error;
  Real epsilon,range;
  int runfor=20;
  int semLesion=0;
  int phoLesion=0;
  float lesion;
  int seed=-1;

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
  
  
  /* build a network, with TIME number of time ticks */
    
  build_model();

  examples=load_examples(patfile,SAMPLES);
  load_weights(net,weightFile);
  
  
  /* lesion it */
  if(phoLesion){
    /*output->clampNoise=lesion;*/
       sever_probabilistically(c6,lesion);
  }
  
  if(semLesion){
    /* sem->clampNoise=lesion; */
    sever_probabilistically(c8,lesion); 
    sever_probabilistically(c9,lesion); 
 } 

  load_sem();

  error=0.0;
  count=1;
  wrongs=0;
  wrongtot=0;
  sse=0.0;

  /* loop for ITER number of times */
  for(i=0;i<examples->numExamples;i++)
    {
      /* get i'th example from exampleset */
      ex=&examples->examples[i];
      get_name(ex->name,name);

      /* do forward propagation */
      crbp_forward(net,ex);
      error=compute_error(net,ex);
      sse+=error;
      wrong=0;

      

      euclid(sem->outputs[SAMPLES-1],euclid_output);
      if(sem->outputs[SAMPLES-1][SEM_FEATURES]> .5)
	strcat(euclid_output,"-PT");
      printf("%s\t",euclid_output);

      for(j=0;j<sem->numUnits-1;j++)
        target[j]=get_value(ex->targets,sem->index,SAMPLES-1,j);

      euclid(target,target_output);
      if(get_value(ex->targets,sem->index,SAMPLES-1,SEM_FEATURES) > .5)
	strcat(target_output,"-PT");

      printf("%s\t",target_output);

      if (strcmp(target_output,euclid_output)> 0)
	{
	  printf("[WRONG]");
	  wrongs++;
	}
      printf("\t(%.2f)\n",error);
    }
  printf("%d wrong\t%.2f mean error\n",wrongs,sse/(float)i);
  return 0;
}
