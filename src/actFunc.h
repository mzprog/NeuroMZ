#ifndef _ACTFUNC_H_
#define _ACTFUNC_H_

#define X 		0
#define SIGMOID 	1
#define TANH 		2
#define ARCTAN 		3
#define SOFTSIGN 	4
#define RELU 		5
#define LRELU 		6
#define SOFTPLUS 	7
#define SINUSOID 	8
#define SINC 		9
#define GAUSSIAN 	10

// double ACTfunc(double x, int flag);
// double ACTderv(double x, int flag);

void * ACTf_Ptr(int flag);
void * ACTd_Ptr(int flag);

double ACTx(double x);
double ACTxDerv(double x);
double ACTsigmoid(double x);
double ACTsigmoidDerv(double x);
double ACTtanh(double x);
double ACTtanhDerv(double x);
double ACTarctan(double x);
double ACTarctanDerv(double x);
double ACTsoftsign(double x);
double ACTsoftsignDerv(double x);
double ACTrelu(double x);
double ACTreluDerv(double x);
double ACTlrelu(double x);
double ACTlreluDerv(double x);
double ACTsoftplus(double x);
double ACTsoftplusDerv(double x);
double ACTsinusoid(double x);
double ACTsinusoidDerv(double x);
double ACTsinc(double x);
double ACTsincDerv(double x);
double ACTgaussian(double x);
double ACTgaussianDerv(double x);


#endif
