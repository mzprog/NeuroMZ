#include "actFunc.h"
#include <math.h>

double ACTfunc(double x, int flag)
{
	switch(flag)
	{
		case x :
			return ACTx(x);
		case BINARY :
			return ACTbin(x);
		case SIGMOID :
			return ACTsigmoid(x);
		case TANH :
			return ACTtanh(x);
		case ARCTAN :
			return ACTarctan(x);
		case SOFTSIGN :
			return ACTsoftsign(x);
		case ISRU :
			return ACTisru(x);
		case RELU :
			return ACTrelu(x);
		case LRELU :
			return ACTlrelu(x);
		case SOFTPLUS :
			return ACTsoftplus(x);
		case SINSOID :
			return ACTsinsoid(x);
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
                case x :
                        return ACTxDerv(x);
                case BINARY :
                        return ACTbinDerv(x);
                case SIGMOID :
                        return ACTsigmoidDerv(x);
                case TANH :
                        return ACTtanhDerv(x);
                case ARCTAN :
                        return ACTarctanDerv(x);
                case SOFTSIGN :
                        return ACTsoftsignDerv(x);
                case ISRU :
                        return ACTisruDerv(x);
                case RELU :
                        return ACTreluDerv(x);
                case LRELU :
                        return ACTlreluDerv(x);
                case SOFTPLUS :
                        return ACTsoftplusDerv(x);
                case SINSOID :
                        return ACTsinsoidDerv(x);
                case SINC :
                        return ACTsincDerv(x);
		case GAUSSIAN :
                        return ACTgaussianDerv(x);
                default :
                        printf("Error: no activation function determined.\n");
                        return 0;
        }
}

