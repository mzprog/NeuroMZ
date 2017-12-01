#include "ann.h"
#include "tools.h"
#include <stdlib.h>
#include <math.h>



double sigmoid(double value){

	return 1/(1 + exp(-value));

}


int INIT_NETWORK(int * layer_number,int layer_size){

	int i,j,k;

	//check if the data from user is valid
	if(layer_size<=0)
		return -1;
	for(i=0;i<layer_size;i++)
		if(layer_number[i]<=0)
			return -1;
	layers_count=layer_size;
	// allocate the layers structures
	//layer= malloc(layer_size* sizeof(struct layers));
	layer= (struct layers*)calloc(layer_size,sizeof(struct layers));

	if(layer==NULL)
		return -1;
	//allocate the neurals for the layers
	for(i=0;i<layer_size;i++)
	{
		layer[i].size=layer_number[i];
		layer[i].neural=(struct neurals *)calloc(layer[i].size+1,sizeof(struct neurals));//i
		
		if(layer[i].neural==NULL)
			return -1;
		if(i==layer_size-1)
			break;

		layer[i].weight=(struct matrix**)calloc(layer[i].size,sizeof(struct matrix *));//i
		
		if(layer[i].weight==NULL)
			return -1;
		for(j=0;j<layer_number[i];j++)
		{
			layer[i].weight[j]=(struct matrix*)calloc(layer_number[i+1],sizeof(struct matrix));//i+1
			if(layer[i].weight[j]==NULL)
				return -1;
			for(k=0;k<layer_number[i+1];k++)//give the weights and the bais random values.
			{
				layer[i].weight[j][k].weightV= 3.0/(rand()%20+1)-1.5;
			}
		}	
		layer[i].neural[j].bais=3.0/(rand()%20+1) -1.5;
	}



	for(j=0;j<layer_number[i];j++)
		layer[i].neural[j].bais=3.0/(rand()%20+1)-1.5;
	output_val=malloc(sizeof(double)*layer[i].size);
	if(output_val==NULL)
		return -1;

	return 0;
}

double * forward(double * data){


	int i,j,k;
	double sum;
	for(i=0;i<layer[0].size;i++)
		layer[0].neural[i].a=data[i];//we put it in the var a because input doesn't need activation functions

	for(k=0;k<layers_count-1;k++)//this loop to all layers.
		for(j=0;j<layer[k+1].size;j++)
		{

			sum=0;
			for(i=0;i<layer[k].size;i++){
				sum+=layer[k].neural[i].a*layer[k].weight[i][j].weightV;//the sum here
			}
			sum+=layer[k+1].neural[j].bais;//finaly add the bais
			layer[k+1].neural[j].x=sum;//then assign it to the x input
			layer[k+1].neural[j].a=sigmoid(layer[k+1].neural[j].x);//at the end assign the value of activation funtion to var a.
		}
	for(i=0;i<layer[k].size;i++)
		output_val[i]=layer[k].neural[i].a;
	return output_val;

}

double  cost_fx(double * target_val){
	double  ret_val,x;
	int i;
	ret_val=0;
	for(i=0;i<layer[layers_count-1].size;i++)
	{
		x=(double) target_val[i]-layer[layers_count-1].neural[i].a;
		ret_val+=x*x;
	}
	return (0.5*ret_val);

}

void backProp(double * target){
				
	int i, j, k,n;//for dimesion of the neurons
	double delta_k,delta_j;//saving the delta 
	//find the bais and  weight delta, and correct it for the last layer
	for(i=0;i<layer[layers_count-1].size;i++){
		delta_k=(target[i]-layer[layers_count-1].neural[i].a)*layer[layers_count-1].neural[i].a*(1 -layer[layers_count-1].neural[i].a);
		//correct the bais
		layer[layers_count-1].neural[i].bais+= learnRate*delta_k;
//here is temporry value for n=0
		for(j=0;j<layer[layers_count-2].size;j++){
			layer[layers_count-2].weight[j][0].weightV+=learnRate*delta_k*layer[layers_count-2].neural[j].a;
			layer[layers_count-2].weight[j][0].deltaW=delta_k;
		}
	}
	//find bais and weight delta for the rest of the layers
	for(k=layers_count-1;k>1;k--)//loop for layers
	{
		for(j=0;j<layer[k-1].size;j++)
		{
			delta_j=0;
			for(n=0;n<layer[k].size;n++)//here new edition
				delta_j+=(layer[k-1].weight[j][n].weightV*layer[k-1].weight[j][n].deltaW);
			delta_j*=(layer[k-1].neural[j].a*(1-layer[k-1].neural[j].a));//final delta j calculation.
			
			//correct the bais
			layer[k-1].neural[j].bais+=(learnRate * delta_j);
			//correct the weights
			for(i=0;i<layer[k-2].size;i++)
			{
				layer[k-2].weight[i][j].weightV+= (learnRate * delta_j *layer[k-2].neural[i].a );
				layer[k-2].weight[i][j].deltaW=delta_j;
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
	head.layers_count=layers_count;
	head.trainNum=steps;
	head.learnRate=learnRate;
	head.conv=conv_value;
	strcpy(head.version,"1.0");

	fptr=fopen(fileName,"w");
	if(fptr==NULL)
		return -1;
	//first add the file header that is the data in general about the network
	fwrite( &head,sizeof(head),1,fptr);

	//preparing for layers element array
	temp=(int *) malloc(layers_count);
	if (temp==NULL){
		fclose(fptr);
		return -1;
	}
	for(i=0;i<layers_count;i++)
		temp[i]=layer[i].size;
	fwrite(temp,sizeof(int),layers_count,fptr);
	//now write the weight of the network
	//find the max
	for(i=0;i<layers_count;i++)
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
	for(k=0;k<layers_count-1;k++){
		for(j=0;j<layer[k].size;j++){
			for(i=0;i<layer[k+1].size;i++)
			{
				//get the 1D array
				tempW[i]=layer[k].weight[j][i].weightV;
			}
			fwrite(tempW,sizeof(double),layer[k+1].size,fptr);
		}
	}
	//now write the bais
	for(k=1;k<layers_count;k++)
	{
		for(i=0;i<layer[k].size;i++)
			tempW[i]=layer[k].neural[i].bais;
		fwrite(tempW,sizeof(double),layer[k].size,fptr);
	}
	//add the train set to the file
	train_set=trnHead;

	while(train_set)
	{
		fwrite(train_set->input,sizeof(double),layer[0].size,fptr);
		fwrite(train_set->output,sizeof(double),layer[layers_count-1].size,fptr);
		train_set=train_set->next;
	}

	free(tempW);
	free(temp);
	fclose(fptr);
	return 0;
}


int loadNet(char * fileName){

	int i, j, k;
	struct fileHead head;
	struct trainSet * train_set;
	int * tempI=0;
	int max=0;
	double * tempD=0;

	FILE * fptr;

	fptr = fopen(fileName,"r");
	if(fptr==NULL)
		return -1;
	//get the head data
	fread(&head,sizeof(head),1,fptr);
	layers_count=head.layers_count;
	steps=head.trainNum;
	learnRate=head.learnRate;
	conv_value=head.conv;

	//get the layer neurons and init network
	tempI=(int *)malloc(sizeof(int)*layers_count);
	if(tempI==NULL){
		fclose(fptr);
		return -1;
	}
	fread(tempI,sizeof(int),layers_count,fptr);
	if(INIT_NETWORK(tempI,layers_count)<0){
		free(tempI);
		fclose(fptr);
		return -1;
	}
	//getting the weights
	//find the max
	for(i=0;i<layers_count;i++)
		if(tempI[i]>max)
			max=tempI[i];
	tempD=(double *) malloc(sizeof(double)*max);
	if(tempD==NULL){
		free(tempI);
		fclose(fptr);
		return -1;
	}
	//clone the weights
	for(k=0;k<layers_count-1;k++){
		for(j=0;j<layer[k].size;j++){
			fread(tempD,sizeof(double),layer[k+1].size,fptr);
			for(i=0;i<layer[k+1].size;i++)
			{
				layer[k].weight[j][i].weightV=tempD[i];
			}
		}
	}
	//clone the bais
	for(k=1;k<layers_count;k++){
		fread(tempD,sizeof(double),layer[k].size,fptr);
		for(j=0;j<layer[k].size;j++)
			layer[k].neural[j].bais=tempD[j];
	}
	//clone the train set
	if(fread(tempD,sizeof(double),layer[0].size,fptr)!=0)
	{
		trnHead=(struct trainSet *) malloc(sizeof(struct trainSet));
		if(trnHead==NULL)
		{
			free(tempD);
			free(tempI);
			fclose(fptr);
			return -1;
		}
		trnHead->input=(double *) malloc(sizeof(double)*layer[0].size);
		if(trnHead->input==NULL)
		{
			free(trnHead);
			trnHead=NULL;
			free(tempD);
			free(tempI);
			fclose(fptr);
			return -1;
		}
		trnHead->output=(double *) malloc(sizeof(double)*layer[layers_count].size);
		if(trnHead->output==NULL)
		{
			free(trnHead->input);
			free(trnHead);
			trnHead=NULL;
			free(tempD);
			free(tempI);
			fclose(fptr);
			return -1;
		}
		for(i=0;i<layer[0].size;i++)
			trnHead->input[i]=tempD[i];
		fread(tempD,sizeof(double),layer[layers_count-1].size,fptr);
		for(i=0;i<layer[layers_count-1].size;i++)
			trnHead->output[i]=tempD[i];
		trnHead->next=NULL;
		train_set=trnHead;
		while(fread(tempD,sizeof(double),layer[0].size,fptr)!=0)
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
			train_set->input=(double *) malloc(sizeof(double)*layer[0].size);
			if(train_set->input==NULL)
			{
				clearTrainSet();
				free(tempD);
				free(tempI);
				fclose(fptr);
				return -1;
			}
			train_set->output=(double * ) malloc(sizeof(double)*layer[layers_count-1].size);
			if(train_set->output==NULL)
			{
				clearTrainSet();
				free(tempD);
				free(tempI);
				fclose(fptr);
				return -1;
			}
			for(i=0;i<layer[0].size;i++)
				train_set->input[i]=tempD[i];
			fread(tempD,sizeof(double),layer[layers_count-1].size,fptr);
			for(i=0;i<layer[layers_count-1].size;i++)
				train_set->output[i]=tempD[i];
		}


	}
	free(tempD);
	free(tempI);
	fclose(fptr);
	return 0;
}

void freeNet(){

	int i,j,k;
	for(k=0;k<layers_count;k++){
		//free the weight matrix colomn
		if(layer[k].weight!=NULL){
			for(j=0;j<layer[k].size;j++)
				free(layer[k].weight[j]);
		}
		//free the raw
		free(layer[k].weight);
		//free the neural
		free(layer[k].neural);

	}

	//free the layers
	free(layer);
	layer=NULL;
	//free the output value array
	free(output_val);
	output_val=NULL;


}
