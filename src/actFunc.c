#include "actFunc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char * func[] = {
    "x", // #           0
    "sigmoid", // #     1
    "tanh", // #        2 
    "arctan", // #      3 
    "softsign", // #    4
    "relu", // #        5
    "lrelu", // #       6
    "softplus", // #    7
    "sinusoid", // #    8
    "sinc", // #        9
    "gaussian" // #     10
} ;


void * ACTf_Ptr(int flag)
{
    switch(flag)
    {
        case X :
                return ACTx;
        case SIGMOID :
                return ACTsigmoid;
        case TANH :
                return ACTtanh;
        case ARCTAN :
                return ACTarctan;
        case SOFTSIGN :
                return ACTsoftsign;
        case RELU :
                return ACTrelu;
        case LRELU :
                return ACTlrelu;
        case SOFTPLUS :
                return ACTsoftplus;
        case SINUSOID :
                return ACTsinusoid;
        case SINC :
                return ACTsinc;
        case GAUSSIAN :
                return ACTgaussian;
        default :
                printf("Error: no activation function determined.\n");
                return NULL;
    }
}

void * ACTd_Ptr(int flag)
{
    switch(flag)
    {
        case X :
                return ACTxDerv;
        case SIGMOID :
                return ACTsigmoidDerv;
        case TANH :
                return ACTtanhDerv;
        case ARCTAN :
                return ACTarctanDerv;
        case SOFTSIGN :
                return ACTsoftsignDerv;
        case RELU :
                return ACTreluDerv;
        case LRELU :
                return ACTlreluDerv;
        case SOFTPLUS :
                return ACTsoftplusDerv;
        case SINUSOID :
                return ACTsinusoidDerv;
        case SINC :
                return ACTsincDerv;
        case GAUSSIAN :
                return ACTgaussianDerv;
        default :
                printf("Error: no activation function determined.\n");
                return NULL;
    }
}


/*
 * This function is wrote as `short` instead of
 * `unsigned short`
 * to allow us to return -1 when we found
 * an error
 */
short getActFlag(char *name)
{
    int i;
    
    for(i = 0; i<ACT_LENGTH; i++)
    {
        if(strcmp(func[i], name) ==0)
        {
            return i;
        }
    }
    return -1;
}

char * getActName(unsigned short flag)
{
    return func[flag];
}





/*
 * activation function and its derivatives 
 * starting from here
 */
double ACTx(double x)
{
	return x;
}

double ACTxDerv(double x)
{
	return 1;
}

double ACTsigmoid(double x)
{
	return (1.0/(1+exp(-x)));
}

double ACTsigmoidDerv(double x)
{
	return ACTsigmoid(x)*(1-ACTsigmoid(x));
}

double ACTtanh(double x)
{
	return (2.0/(1+exp(-2*x))-1);
}
double ACTtanhDerv(double x)
{
	return 1-ACTtanh(x)*ACTtanh(x);
}

double ACTarctan(double x)
{
	return atan(x);
}

double ACTarctanDerv(double x)
{
	return 1.0/(x*x+1);
}

double ACTsoftsign(double x)
{
	return x/(1+abs(x));
}
double ACTsoftsignDerv(double x)
{
	return 1.0/((1+abs(x))*(1+abs(x)));
}

double ACTrelu(double x)
{
	if(x<0)
		return 0;
	else
		return x;
}
double ACTreluDerv(double x)
{
	if(x<0)
		return 0;
	else 
		return 1;
}

double ACTlrelu(double x)
{
	if(x<0)
		return 0.01*x;
	else
		return x;
}
double ACTlreluDerv(double x)
{
	if(x<0)
		return 0.01;
	else
		return 1;
}

double ACTsoftplus(double x)
{
	return log(1+exp(x));
}
double ACTsoftplusDerv(double x)
{
	return 1.0/(1+exp(x));
}

double ACTsinusoid(double x)
{
	return sin(x);
}
double ACTsinusoidDerv(double x)
{
	return cos(x);
}

double ACTsinc(double x)
{
	if(x==0)
		return 1;
	else
		return sin(x)/x;
}
double ACTsincDerv(double x)
{
	if(x==0)
		return 0;
	else
		return cos(x)/x - sin(x)/(x*x);
}

double ACTgaussian(double x)
{
	return exp(-x*x);
}
double ACTgaussianDerv(double x)
{
	return -2.0*exp(-x*x);
}
