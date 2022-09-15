#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include <mikenet/simulator.h>

#include "simconfig.h"
#include "model.h"
#include "model.c"

#define REP 500

int seed=0;

FILE *finfo=NULL;

#define ITERS 5000001

int start=1;

void get_name(char *tag, char *name)
{
  char *p;
  p=strstr(tag,"Word:");
  p+= 5;
  p=strtok(p," \t\n");
  strcpy(name,p);
}

int train()
{
  Example *ex;
  int i,j;
  Real dice,e;
  Real speaking_err=0,hearing_err=0,repeating_err=0,gen_err;
  Real diverge;
  char fn[255];
  int save=1,count=1;
  int hear_count=0,speak_count=0,rep_count=0,gen_count=0;
  int saveCount= 50000;
  char wordname[255];

  for(i=start;i<=start+ITERS;i++)
    {
      dice=mikenet_random();

      if (dice < .4) {
	ex=get_random_example(hearing_examples);
	crbp_forward(net,ex);
	crbp_compute_gradients(net,ex);
	crbp_update_taos(net);
	crbp_apply_deltas(net);
	hear_count++;
	e=compute_error(net,ex);
	hearing_err += e;
      }
      else if (dice < .8) {
	ex=get_random_example(speaking_examples);
	crbp_forward(net,ex);
	crbp_compute_gradients(net,ex);
        crbp_update_taos(net);
        crbp_apply_deltas(net);
        speak_count++;
        e=compute_error(net,ex);
        speaking_err += e;
      }

      else if (dice < .9) {
	ex=get_random_example(repeating_examples);
	crbp_forward(net,ex);
	crbp_compute_gradients(net,ex);
	crbp_update_taos(net);
	crbp_apply_deltas(net);
	rep_count++;
	e=compute_error(net,ex);
	repeating_err += e;
      }
      else  {
	ex=get_random_example(gen_examples);
	crbp_forward(net,ex);
	crbp_compute_gradients(net,ex);
	crbp_update_taos(net);
	crbp_apply_deltas(net);
	gen_count++;
	e=compute_error(net,ex);
	gen_err += e;
      }


      if (count==REP)
	{
	  fflush(finfo);
	  printf("iter %d\ts: %.4f\th: %.4f\tr: %.4f\tg: %.4f",i,
		 speaking_err/(float)speak_count,
		 hearing_err/(float)hear_count,
		 repeating_err/(float)rep_count,
		 gen_err/(float)gen_count);
	  speaking_err=0.0;
	  hearing_err=0.0;
	  gen_err=0.0;
	  repeating_err=0.0;
	  hear_count=0;
	  rep_count=0;
	  gen_count=0;
	  speak_count=0;
	  count=1;

	  /* print out some info on hidden unit stress */
	  diverge=0;
	  for(j=0;j<hidden->numUnits;j++)
	    diverge+= fabs(0.5 - hidden->outputs[SAMPLES-1][j]);
	  diverge/=(float)hidden->numUnits;
	  printf("\t%f\n",diverge);
	}
      else count++;
      
      if (save==saveCount)
	{
	  sprintf(fn,"%d_weights",i);
	  save_weights(net,fn);
	  save=1;
	}
      else save++;
    }

}

int main(int argc,char *argv[])
{
  int i;
  char loadWeights[255];
  char fn[255];
  char infoFile[255];
  char  sPatFile[255],hPatFile[255],rPatFile[255],gPatFile[256];
  float epsilon=0.05;
  setbuf(stdout,NULL);

  loadWeights[0]=0;
  sPatFile[0]=0;
  hPatFile[0]=0;
  rPatFile[0]=0;
  infoFile[0]=0;
  announce_version();
  for(i=1;i<argc;i++)
    {
      printf("arg %s\n",argv[i]);
      if (strcmp(argv[i],"-seed")==0)
	{
	  seed=atoi(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-info")==0)
	{
	  strcpy(infoFile,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-spat")==0)
	{
	  strcpy(sPatFile,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-rpat")==0)
	{
	  strcpy(rPatFile,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-hpat")==0)
	{
	  strcpy(hPatFile,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-gpat")==0)
	{
	  strcpy(gPatFile,argv[i+1]);
	  i++;
	}

      else if (strcmp(argv[i],"-load")==0)
	{
	  strcpy(loadWeights,argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-epsilon")==0)
	{
	  epsilon=atof(argv[i+1]);
	  i++;
	}
      else if (strcmp(argv[i],"-start")==0)
	{
	  start=atoi(argv[i+1]);
	  i++;
	}
      else
	{
	  fprintf(stderr,"unknown option: %s\n",argv[i]);
	  exit(-1);
	}
    }
  /*  
      if (infoFile[0]==0)
      Error0("Need to specify info file with -info option");  
      finfo=fopen(infoFile,"a");
*/
  default_epsilon=epsilon;
  if (sPatFile[0]==0 || hPatFile[0]==0 || rPatFile[0]==0 || gPatFile[0]==0)
    Error0("Need to specify training files with -pat option");

  if(seed==0)
//    seed=getpid(); // doesn't work on  MacOS
  seed = 666; // the seed of the beast

  printf("seed %d\t epsilon %f\n",seed,epsilon);
  system("date");
  system("hostname");
  mikenet_set_seed(seed);
 
  /* build a network, with SAMPLES number of time ticks */
  build_model();
  printf("start %d\n",start); 
  if (start==1)
    {
      sprintf(fn,"s%d_init_weights",seed);
      save_weights(net,fn);
    }
  else 
    {
      load_weights(net,loadWeights);
      printf("loaded weight file %s\n",loadWeights);
    }
  /* load in our example set */
  fprintf(stderr,"loading training file  %s\n",sPatFile);
  speaking_examples=load_examples(sPatFile,SAMPLES);
  hearing_examples=load_examples(hPatFile,SAMPLES);
  repeating_examples=load_examples(rPatFile,SAMPLES);
  gen_examples=load_examples(gPatFile,SAMPLES);

  fprintf(stderr,"done\n");

  train();
  sprintf(fn,"s%d_final_weights",seed);
  save_weights(net,fn);
  printf("finished\n");
  fclose(finfo);
  system("date");
  return 0;
}
