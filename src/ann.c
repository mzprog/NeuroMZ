#include "ann.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "neuromz.h"
#include "actFunc.h"


NEUROMZ_data * neuromz = NULL;

int INIT_NETWORK(int * layer_number,uint16 *actf,int layer_size){

	int i,j,k;

	//check if the data from user is valid
	if(layer_size<=0)
		return -1;
	for(i=0;i<layer_size;i++)
		if(layer_number[i]<=0)
			return -1;
        
    neuromz = (NEUROMZ_data *) calloc(1,sizeof(NEUROMZ_data));
    if(neuromz == NULL)
    {
        return -1;
    }
    //set some vars and the rest are NULL
    neuromz->conv_value = 0.005;
    neuromz->learnRate = 0.1;
    
        
	neuromz->layers_count=layer_size;
	// allocate the layers structures
	//layer= malloc(layer_size* sizeof(struct layers));
	neuromz->layer= (struct layers*)calloc(layer_size,sizeof(struct layers));

	if(neuromz->layer==NULL)
		return -1;
	//allocate the neurals for the layers
	for(i=0;i<layer_size;i++)
	{
		neuromz->layer[i].size=layer_number[i];
		neuromz->layer[i].neural=(struct neurals *)calloc(neuromz->layer[i].size+1,sizeof(struct neurals));//i
		neuromz->layer[i].ACT_FX=actf[i];//use sigmoid function as default		
		neuromz->layer[i].actf = ACTf_Ptr(actf[i]);
		neuromz->layer[i].actd = ACTd_Ptr(actf[i]);
		if(neuromz->layer[i].neural==NULL)
			return -1;
		if(i==layer_size-1)
			break;

		neuromz->layer[i].weight=(struct matrix**)calloc(neuromz->layer[i].size,sizeof(struct matrix *));//i
		
		if(neuromz->layer[i].weight==NULL)
			return -1;
		for(j=0;j<layer_number[i];j++)
		{
			neuromz->layer[i].weight[j]=(struct matrix*)calloc(layer_number[i+1],sizeof(struct matrix));//i+1
			if(neuromz->layer[i].weight[j]==NULL)
				return -1;
			for(k=0;k<layer_number[i+1];k++)//give the weights and the bais random values.
			{
				neuromz->layer[i].weight[j][k].weightV= 3.0/(rand()%20+1)-1.5;
			}
		}	
		neuromz->layer[i].neural[j].bais=3.0/(rand()%20+1) -1.5;
	}



	for(j=0;j<layer_number[i];j++)
		neuromz->layer[i].neural[j].bais=3.0/(rand()%20+1)-1.5;
	neuromz->output_val=malloc(sizeof(double)*neuromz->layer[i].size);
	if(neuromz->output_val==NULL)
		return -1;

	return 0;
}

double * forward(double * data){

	int i,j,k;
	double sum;
	for(i=0;i<neuromz->layer[0].size;i++)
		neuromz->layer[0].neural[i].a=data[i];//we put it in the var a because input doesn't need activation functions
	for(k=0;k<neuromz->layers_count-1;k++)//this loop to all layers.
		for(j=0;j<neuromz->layer[k+1].size;j++)
		{
			sum=0;
			for(i=0;i<neuromz->layer[k].size;i++){
				sum+=neuromz->layer[k].neural[i].a*neuromz->layer[k].weight[i][j].weightV;//the sum here
			}
			sum+=neuromz->layer[k+1].neural[j].bais;//finaly add the bais
			neuromz->layer[k+1].neural[j].x=sum;//then assign it to the x input
			neuromz->layer[k+1].neural[j].a = neuromz->layer[k+1].actf(neuromz->layer[k+1].neural[j].x);

		}
	for(i=0;i<neuromz->layer[k].size;i++)
		neuromz->output_val[i]=neuromz->layer[k].neural[i].a;
	return neuromz->output_val;

}

double  cost_fx(double * target_val){
	double  ret_val,x;
	int i;
	ret_val=0;
	for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++)
	{
		x=(double) target_val[i]-neuromz->layer[neuromz->layers_count-1].neural[i].a;
		ret_val+=x*x;
	}
	return (0.5*ret_val);

}

void backProp(double * target){
				
	int i, j, k,n;//for dimesion of the neurons
	double delta_k,delta_j;//saving the delta 
	//find the bais and  weight delta, and correct it for the last layer
	for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++){
		delta_k = (target[i]-neuromz->layer[neuromz->layers_count-1].neural[i].a) * 
                    neuromz->layer[neuromz->layers_count-1].actd(neuromz->layer[neuromz->layers_count-1].neural[i].x);
                    //ACTderv(neuromz->layer[neuromz->layers_count-1].neural[i].x,neuromz->layer[neuromz->layers_count-1].ACT_FX);
		
		//correct the bais
		neuromz->layer[neuromz->layers_count-1].neural[i].bais+= neuromz->learnRate*delta_k;
//here is temporry value for n=0
		for(j=0;j<neuromz->layer[neuromz->layers_count-2].size;j++){
			neuromz->layer[neuromz->layers_count-2].weight[j][0].weightV+=neuromz->learnRate*delta_k*neuromz->layer[neuromz->layers_count-2].neural[j].a;
			neuromz->layer[neuromz->layers_count-2].weight[j][0].deltaW=delta_k;
		}
	}
	//find bais and weight delta for the rest of the layers
	for(k=neuromz->layers_count-1;k>1;k--)//loop for layers
	{
		for(j=0;j<neuromz->layer[k-1].size;j++)
		{
			delta_j=0;
			for(n=0;n<neuromz->layer[k].size;n++)//here new edition
				delta_j+=(neuromz->layer[k-1].weight[j][n].weightV*neuromz->layer[k-1].weight[j][n].deltaW);
			//delta_j*=ACTderv(neuromz->layer[k-1].neural[j].x,neuromz->layer[k-1].ACT_FX);//final gelta j calculation.
			delta_j *= neuromz->layer[k-1].actd(neuromz->layer[k-1].neural[j].x);
			//correct the bais
			neuromz->layer[k-1].neural[j].bais+=(neuromz->learnRate * delta_j);
			//correct the weights
			for(i=0;i<neuromz->layer[k-2].size;i++)
			{
				neuromz->layer[k-2].weight[i][j].weightV+= (neuromz->learnRate * delta_j *neuromz->layer[k-2].neural[i].a );
				neuromz->layer[k-2].weight[i][j].deltaW=delta_j;
			}
		}
	}
}

//////////////////////////////////////////////////////////
//this part of the file is for saving neural network	//
//in a file						//
//and then open it					//
//and the option for closing the network		//
//////////////////////////////////////////////////////////


int saveNet(char * fileName){

	int i,j,k;
	int max=0;
	int * temp=0;
	double * tempW=0;
	struct fileHead head;
	struct trainSet * train_set;
	FILE * fptr;
    
    if(neuromz == NULL)
    {
        printf("Error: no network opened in memory.\n");
		return -1;
    }
    
	head.layers_count=neuromz->layers_count;
	head.trainNum=neuromz->steps;
	head.learnRate=neuromz->learnRate;
	head.conv=neuromz->conv_value;
	strcpy(head.version,"1.1");

	fptr=fopen(fileName,"w");
	if(fptr==NULL)
		return -1;
	//first add the file header that is the data in general about the network
	fwrite( &head,sizeof(head),1,fptr);

	//preparing for layers element array
	temp=(int *) malloc(neuromz->layers_count);
	if (temp==NULL){
		fclose(fptr);
		return -1;
	}
	for(i=0;i<neuromz->layers_count;i++)
		temp[i]=neuromz->layer[i].size;
	fwrite(temp,sizeof(int),neuromz->layers_count,fptr);
	
	//prepare the activation function element array
	for(i=0;i<neuromz->layers_count;i++)
		temp[i]=neuromz->layer[i].ACT_FX;
	fwrite(temp,sizeof(int),neuromz->layers_count,fptr);

	//now write the weight of the network
	//find the max
	for(i=0;i<neuromz->layers_count;i++)
		if(temp[i]>max)
			max=temp[i];

	tempW=(double *)malloc(sizeof(double)*max);
	if(tempW==NULL)
	{
		free(temp);
		fclose(fptr);
		return -1;
	}

	//prepare the array and write to the file
	for(k=0;k<neuromz->layers_count-1;k++){
		for(j=0;j<neuromz->layer[k].size;j++){
			for(i=0;i<neuromz->layer[k+1].size;i++)
			{
				//get the 1D array
				tempW[i]=neuromz->layer[k].weight[j][i].weightV;
			}
			fwrite(tempW,sizeof(double),neuromz->layer[k+1].size,fptr);
		}
	}
	//now write the bais
	for(k=1;k<neuromz->layers_count;k++)
	{
		for(i=0;i<neuromz->layer[k].size;i++)
			tempW[i]=neuromz->layer[k].neural[i].bais;
		fwrite(tempW,sizeof(double),neuromz->layer[k].size,fptr);
	}
	//add the train set to the file
	train_set = neuromz->trnHead;

	while(train_set)
	{
		fwrite(train_set->input,sizeof(double),neuromz->layer[0].size,fptr);
		fwrite(train_set->output,sizeof(double),neuromz->layer[neuromz->layers_count-1].size,fptr);
		train_set=train_set->next;
	}

	free(tempW);
	free(temp);
	fclose(fptr);
	return 0;
}


int loadNet(char * fileName){
puts("a");
	int i, j, k;
	struct fileHead head;
	struct trainSet * train_set;
    uint16 * act;
	int * tempI=0;
	int max=0;
	double * tempD=0;

    FILE * fptr;
    neuromz = (NEUROMZ_data *) calloc(1,sizeof(NEUROMZ_data));
    if(neuromz == NULL)
    {
        printf("Error: Unable to open in memory.\n");
		return -1;
    }

    fptr = fopen(fileName,"r");
	if(fptr==NULL)
		return -1;
	//get the head data
	fread(&head,sizeof(head),1,fptr);
	neuromz->layers_count=head.layers_count;
	neuromz->steps=head.trainNum;
	neuromz->learnRate=head.learnRate;
	neuromz->conv_value=head.conv;

    //get the layer neurons and init network
	tempI=(int *)malloc(sizeof(int)*neuromz->layers_count);
	if(tempI==NULL){
		fclose(fptr);
		return -1;
	}
	
	act = (uint16 *) calloc(neuromz->layers_count,sizeof(uint16));
    if(act == NULL)
    {
        fclose(fptr);
        return -1;
    }
    
    for(i = 0; i<neuromz->layers_count; i++)
    {
        act[i] = SIGMOID;
    }
	
	fread(tempI,sizeof(int),neuromz->layers_count,fptr);
	if(INIT_NETWORK(tempI, act, neuromz->layers_count)<0){
        free(act);
		free(tempI);
		fclose(fptr);
		return -1;
	}
	
	free(act);

    //get the activation function flags
	if(strcmp(head.version,"1.1")==0)
	{
		fread(tempI,sizeof(int),neuromz->layers_count,fptr);
		for(i=0;i<neuromz->layers_count;i++)
			neuromz->layer[i].ACT_FX=tempI[i];
            neuromz->layer[i].actd = ACTd_Ptr(tempI[i]);
            neuromz->layer[i].actf = ACTf_Ptr(tempI[i]);
	}
	else if(strcmp(head.version,"1.0")==0)
	{
		//as defualt for v 1.0 that the default activation function is sigmoid
		for(i=0;i<neuromz->layers_count;i++)
        {
			neuromz->layer[i].ACT_FX=SIGMOID;
            neuromz->layer[i].actd = ACTd_Ptr(SIGMOID);
            neuromz->layer[i].actf = ACTf_Ptr(SIGMOID);
        }
	}

	//getting the weights
	//find the max
	for(i=0;i<neuromz->layers_count;i++)
		if(tempI[i]>max)
			max=tempI[i];
	tempD=(double *) malloc(sizeof(double)*max);
	if(tempD==NULL){
		free(tempI);
		fclose(fptr);
		return -1;
	}

	//clone the weights
	for(k=0;k<neuromz->layers_count-1;k++){
		for(j=0;j<neuromz->layer[k].size;j++){
			fread(tempD,sizeof(double),neuromz->layer[k+1].size,fptr);
			for(i=0;i<neuromz->layer[k+1].size;i++)
			{
				neuromz->layer[k].weight[j][i].weightV=tempD[i];
			}
		}
	}

	//clone the bais
	for(k=1;k<neuromz->layers_count;k++){
		fread(tempD,sizeof(double),neuromz->layer[k].size,fptr);
		for(j=0;j<neuromz->layer[k].size;j++)
			neuromz->layer[k].neural[j].bais=tempD[j];
	}

	//clone the train set
	if(fread(tempD,sizeof(double),neuromz->layer[0].size,fptr)!=0)
	{
		neuromz->trnHead=(struct trainSet *) malloc(sizeof(struct trainSet));
		if(neuromz->trnHead==NULL)
		{
			free(tempD);
			free(tempI);
			fclose(fptr);
			return -1;
		}
		neuromz->trnHead->input=(double *) malloc(sizeof(double)*neuromz->layer[0].size);
		if(neuromz->trnHead->input==NULL)
		{
			free(neuromz->trnHead);
			neuromz->trnHead=NULL;
			free(tempD);
			free(tempI);
			fclose(fptr);
			return -1;
		}
		neuromz->trnHead->output=(double *) malloc(sizeof(double)*neuromz->layer[neuromz->layers_count].size);
		if(neuromz->trnHead->output==NULL)
		{
			free(neuromz->trnHead->input);
			free(neuromz->trnHead);
			neuromz->trnHead=NULL;
			free(tempD);
			free(tempI);
			fclose(fptr);
			return -1;
		}
		for(i=0;i<neuromz->layer[0].size;i++)
			neuromz->trnHead->input[i]=tempD[i];
		fread(tempD,sizeof(double),neuromz->layer[neuromz->layers_count-1].size,fptr);
		for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++)
			neuromz->trnHead->output[i]=tempD[i];
		neuromz->trnHead->next=NULL;
		train_set=neuromz->trnHead;
		while(fread(tempD,sizeof(double),neuromz->layer[0].size,fptr)!=0)
		{
			train_set->next=(struct trainSet *) malloc(sizeof(struct trainSet));
			if(train_set->next==NULL)
			{
				clearTrainSet();
				free(tempD);
				free(tempI);
				fclose(fptr);
				return -1;
			}
			train_set=train_set->next;
			train_set->next=NULL;
			train_set->input=(double *) malloc(sizeof(double)*neuromz->layer[0].size);
			if(train_set->input==NULL)
			{
				clearTrainSet();
				free(tempD);
				free(tempI);
				fclose(fptr);
				return -1;
			}
			train_set->output=(double * ) malloc(sizeof(double)*neuromz->layer[neuromz->layers_count-1].size);
			if(train_set->output==NULL)
			{
				clearTrainSet();
				free(tempD);
				free(tempI);
				fclose(fptr);
				return -1;
			}
			for(i=0;i<neuromz->layer[0].size;i++)
				train_set->input[i]=tempD[i];
			fread(tempD,sizeof(double),neuromz->layer[neuromz->layers_count-1].size,fptr);
			for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++)
				train_set->output[i]=tempD[i];
		}


	}

	free(tempD);
	free(tempI);
	fclose(fptr);
	return 0;
}

void freeNet(){

	int j,k;
	for(k=0;k<neuromz->layers_count;k++){
		//free the weight matrix colomn
		if(neuromz->layer[k].weight!=NULL){
			for(j=0;j<neuromz->layer[k].size;j++)
				free(neuromz->layer[k].weight[j]);
		}
		//free the raw
		free(neuromz->layer[k].weight);
		//free the neural
		free(neuromz->layer[k].neural);

	}

	//free the layers
	free(neuromz->layer);
	neuromz->layer=NULL;
	//free the output value array
	free(neuromz->output_val);
	neuromz->output_val=NULL;

    if(neuromz->filename != NULL)
    {
        free(neuromz->filename);
        neuromz->filename =NULL;
    }
    
    free(neuromz);
    neuromz = NULL;
}
