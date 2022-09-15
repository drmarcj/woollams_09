#include <stdio.h>
#include <math.h>
#include <mikenet/simulator.h>

#include "model.h"
#include "simconfig.h"

Net *net;
Group *sem,*hidden,*phonIn,*phonOut,*pho_cleanup,*sem_cleanup,*bias;
Connections *c1,*c2,*c3,*c4,*c5,*c6,*c7,*c8,*c9,*c10,*c11,*c12,*c13;
ExampleSet *speaking_examples,*hearing_examples,*gen_examples,*repeating_examples;

build_model() {
  float range;

  net=create_net(SAMPLES);
  net->integrationConstant=(float)SECONDS/(float)SAMPLES;

  default_errorComputation=CROSS_ENTROPY_ERROR;
  default_tai=1;
  default_errorRamp=RAMP_ERROR;
  default_activationType=LOGISTIC_ACTIVATION;
    default_errorRadius=0.1;

  phonIn=init_group("PhonIn",PHO_FEATURES*PHO_SLOTS,SAMPLES);
  phonOut=init_group("PhonOut",PHO_FEATURES*PHO_SLOTS,SAMPLES);
  hidden = init_group("Hidden",300,SAMPLES);
  sem=init_group("Sem",SEM_FEATURES+1,SAMPLES);
  pho_cleanup=init_group("PhoCleanup",50,SAMPLES);
  sem_cleanup=init_group("SemCleanup",50,SAMPLES);
  bias = init_bias(1.0,SAMPLES);

  bind_group_to_net(net,phonIn);
  bind_group_to_net(net,phonOut);
  bind_group_to_net(net,sem);
  bind_group_to_net(net,hidden);
  bind_group_to_net(net,sem_cleanup);
  bind_group_to_net(net,pho_cleanup);
  bind_group_to_net(net,bias);


  c1=connect_groups(phonIn,hidden);
  c2=connect_groups(hidden,sem);
  c3=connect_groups(sem,hidden);
  c4=connect_groups(sem,sem_cleanup);
  c5=connect_groups(sem_cleanup,sem);
  c6=connect_groups(hidden,phonOut);
  c7=connect_groups(phonOut,pho_cleanup);
  c8=connect_groups(pho_cleanup,phonOut);

  c9=connect_groups(bias,hidden);
  c10=connect_groups(bias,phonOut);
  c11=connect_groups(bias,sem);
  c12=connect_groups(bias,pho_cleanup);
  c13=connect_groups(bias,sem_cleanup);

  bind_connection_to_net(net,c1);
  bind_connection_to_net(net,c2);
  bind_connection_to_net(net,c3);
  bind_connection_to_net(net,c4);
  bind_connection_to_net(net,c5);
  bind_connection_to_net(net,c6);
  bind_connection_to_net(net,c7);
  bind_connection_to_net(net,c8);
  bind_connection_to_net(net,c9);
  bind_connection_to_net(net,c10);
  bind_connection_to_net(net,c11);
  bind_connection_to_net(net,c12);
  bind_connection_to_net(net,c13);



  range=0.1;
  randomize_connections(c1,range);
  randomize_connections(c2,range);
  randomize_connections(c3,range);
  randomize_connections(c4,range);
  randomize_connections(c5,range);
  randomize_connections(c6,range);
  randomize_connections(c7,range);
  randomize_connections(c8,range);
  randomize_connections(c9,range);
  randomize_connections(c10,range);
  randomize_connections(c11,range);
  randomize_connections(c12,range);
  randomize_connections(c13,range);
}

