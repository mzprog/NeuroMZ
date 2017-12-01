#ifndef _ANN_H_
#define _ANN_H_

#define uint16 unsigned short

//define the neurals structure
struct neurals{
	double x;//the input 
	double a;//the output
	double bais;
};
//define struct for both weight and deltaW
struct matrix{
	double weightV;
	double deltaW;
};

//define the layers structure
struct layers{
	struct neurals * neural;//pointer to array of neural for a single layer
	uint16 size;
	struct matrix ** weight;//the weight's matrix for this layer and the next one


};

//data and struct for file saving and openning file
struct fileHead{
	char version[10];
	int layers_count;
	unsigned long trainNum;
	double learnRate;
	double conv;
		
};

double *  temp=0;//we should allocate as the bigest layer neurons.


//aditional global varaibles
struct layers * layer=0;
uint16 layers_count;
unsigned long steps=0;
double * output_val=0;
double learnRate = 0.1;//we can change it later by the main function
int INIT_NETWORK(int *layer_number,int layer_size);//preparing the neural network

double * forward(double * data);//entring an array of double and the result is the array of the output

double cost_fx(double * target_val);//the cost function

void backProp(double * target);

double sigmoid(double value);

int saveNet(char * fileName);

int loadNet(char * fileName);
void freeNet();
#endif
