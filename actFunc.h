#ifndef _ACTFUNC_H_
#define _ACTFUNC_H_

#define X 		0
#define BINARY 		1
#define SIGMOID 	2
#define TANH 		3
#define ARCTAN 		4
#define SOFTSIGN 	5
#define ISRU 		6
#define RELU 		7
#define LRELU 		8
#define SOFTPLUS 	9
#define SINUSOID 	10
#define SINC 		11
#define GAUSSIAN 	12

double ACTfunc(double x, int flag);
double ACTderv(double x, int flag);

double ACTx(double x);
double ACTxDerv(double x);
double ACTbin(double x);
double ACTbinDerv(double x);
double ACTsigmoid(double x);
double ACTsigmoidDerv(double x);
double ACTtanh(double x);
double ACTtanhDrev(double x);
double ACTarctan(double x);
double ACTarctanDerv(double x);
double ACTsoftsign(double x);
double ACTsoftsignDerv(double x);
double ACTisru(double x);
double ACTisruDerv(double x);
double ACTrelu(double x);
double ACTreluDerv(double x);
double ACTlrelu(double x);
double ACTlreluDerv(double x);
double ACTsoftplus(double x);
double ACTsoftplusDerv(double x);
double ACTsinusoid(double x);
double ACTsinsoidDerv(double x);
double ACTsinc(double x);
double ACTsincDerv(double x);
double ACTgaussain(double x);
double ACTgaussainDerv(double x);


#endif
