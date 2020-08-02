#include "actFunc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
/*
double ACTfunc(double x, int flag)
{
	switch(flag)
	{
		case X :
			return ACTx(x);
		case SIGMOID :
			return ACTsigmoid(x);
		case TANH :
			return ACTtanh(x);
		case ARCTAN :
			return ACTarctan(x);
		case SOFTSIGN :
			return ACTsoftsign(x);
		case RELU :
			return ACTrelu(x);
		case LRELU :
			return ACTlrelu(x);
		case SOFTPLUS :
			return ACTsoftplus(x);
		case SINUSOID :
			return ACTsinusoid(x);
		case SINC :
			return ACTsinc(x); 
		case GAUSSIAN :
			return ACTgaussian(x);
		default :
			printf("Error: no activation function determined.\n");
			return 0;
	}
}


double ACTderv(double x, int flag)
{
        switch(flag)
        {
                case X :
                        return ACTxDerv(x);
                case SIGMOID :
                        return ACTsigmoidDerv(x);
                case TANH :
                        return ACTtanhDerv(x);
                case ARCTAN :
                        return ACTarctanDerv(x);
                case SOFTSIGN :
                        return ACTsoftsignDerv(x);
                case RELU :
                        return ACTreluDerv(x);
                case LRELU :
                        return ACTlreluDerv(x);
                case SOFTPLUS :
                        return ACTsoftplusDerv(x);
                case SINUSOID :
                        return ACTsinusoidDerv(x);
                case SINC :
                        return ACTsincDerv(x);
		case GAUSSIAN :
                        return ACTgaussianDerv(x);
                default :
                        printf("Error: no activation function determined.\n");
                        return 0;
        }
}
*/


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
