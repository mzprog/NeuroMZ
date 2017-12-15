#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"
#include "neuromz.h"
#include "config.h"

char QUIT=0;//for quit the command function. 

extern struct trainSet * trnHead;
extern double *  temp;//we should allocate as the bigest layer neurons.
extern char * filename;
extern double conv_value;
extern struct layers * layer;
extern uint16 layers_count;
extern unsigned long steps;
extern double * output_val;
extern double learnRate;//we can change it later by the main function


void clearWords(struct words *w){
	
	while(w->next){
		w=w->next;
	}
	while(w->prv){
		w=w->prv;
		free(w->prv);
	}
	memset(w->t,0,32);
	w->next=NULL;
}


void checkLine(char * line){

	int i, j=0, len,count_w=0;
	char tmp[32];
	struct words * word=(struct words *)malloc(sizeof(struct words));
	word->prv=NULL;
	word->next=NULL;
	len=strlen(line);
	memset(tmp,0,32);

	for(i=0;i<len+1;i++)
	{
		if(line[i]==' ' || line[i]=='\0')
		{
			if(strlen(tmp)==0)
				continue;
			count_w++;
			newNode(tmp,word);
			memset(tmp,0,32);
			j=0;
		}
		else{
			tmp[j++]=line[i];
		}

	}

	while(word->prv!=NULL){
		word=word->prv;
	}
	
	//now we star checking the line
	if(strcmp(word->t,"exit")==0)
	{
		if(layer!=0)
			freeNet();
		QUIT=1;
	}
	else if(strcmp(word->t,"load")==0)
	{
		loadFile(word,count_w);
	}
	else if(strcmp(word->t,"save")==0)
	{
		saveFile(word,count_w);
	}
	else if(strcmp(word->t,"train")==0)
	{
		doTrain(word,count_w);
	}
	else if(strcmp(word->t,"test")==0)
	{
		doForward(word,count_w);
	}
	else if(strcmp(word->t,"new")==0)
	{
		createNet(word,count_w);
	}
	else if(strcmp(word->t,"clear")==0)
	{
		freeNet();
		clearTrainSet();
		if(filename!=NULL){
			free(filename);
			filename=NULL;
		}
	}
	else if(strcmp(word->t,"learnrate")==0)
	{
		learn_rate(word,count_w);
	}
	else if(strcmp(word->t,"conv")==0)
	{
		convarge(word,count_w);
	}
	else if(strcmp(word->t,"help")==0 || strcmp(word->t,"-h")==0)
	{
		printHelp(word,count_w);
	}
	else if(strcmp(word->t,"show")==0)
	{
		showDet();
	}
	else if(strcmp(word->t,"version")==0 || strcmp(word->t,"-v")==0)
	{
		versionP();
	}
	else if(strcmp(word->t,"rmset")==0)
	{
		if(layer==NULL)
		{
			printf("Error: no network to clear its trainning set.\n");
		}
		else
		{
			clearTrainSet();
		}
	}
	else
		printError();

	//clearing the linked list
	while(word->next){
		word=word->next;
	}
	while(word->prv){
		word=word->prv;
		free(word->next);
	}
	free(word);//the last remainning list

}

void newNode(char * txt, struct words *w){

	struct words *tx;
	if(w->prv==NULL && w->next==NULL)//the first one 
	{
		
		strcpy(w->t,txt);
		//w->next=(struct words *)malloc(sizeof(struct words));
		w->next=(struct words *)calloc(1,sizeof(struct words));
		tx=w->next;
		tx->prv=w;
		tx->next=NULL;
	}else
	{
		
		while(w->next){
			w=w->next;
		}
		strcpy(w->t,txt);
		//w->next=(struct words *)malloc(sizeof(struct words));
		w->next=(struct words *)calloc(1,sizeof(struct words));
		tx=w->next;
		tx->prv=w;
		tx->next=NULL;
	}
}



void printError(){
	
	printf("command not found\nyou can type \"-h\" or \"help\".\n");
}

void saveFile(struct words * w, int count){

	char *name;

	if(layer==NULL){
		printf("Error: no data found to save.\n");
		return;
	}

	if(count==1){
		if(filename==0)
		{
			printf("Error: should declare a file name.\n");
			return;
		}else
		{
			name=filename;
		}
	}
	else if(count==2)
	{
		w=w->next;
		name=w->t;
		if(filename!=NULL)
		{
			free(filename);
			filename=NULL;
		}
		filename=(char *)malloc(sizeof(char)*(strlen(name)+1));
		if(filename==NULL){
			printf("Error: unknown error in memory.\n");
		}
		strcpy(filename,name);
	}
	else{
		printf("Error: invalid data.\n");
		return;
	}
	if(saveNet(name)<0){
		printf("Error: can\'t save the file.\n");
		return;
	}
	
}

void loadFile(struct words * w ,int count){
	if(layer != NULL || filename != NULL){
		printf("Error: Can\'t load file, should clear current network before.\n");
		return;
	}
	if(count<2){
		printf("Error: please enter your file name.\n");
		return;
	}else if(count > 2){
		printf("Error: Too much data.\n");
		return;
	}
	w=w->next;
	if(loadNet(w->t)<0){
		printf("Error: This file is not found or have an error in memory.\n");
	}
	else
	{	
		filename=(char *)malloc(sizeof(char)*(strlen(w->t)+1));
		if(filename==NULL){
			printf("Error: unknown error in memory.\n");
			freeNet();
			return;
		}
		strcpy(filename,w->t);
	}

}

int createNet(struct words * w,int count){
	
	int i;
	int *l;
	
	if(layer!=NULL)
	{
		printf("Error: should clear current network brfore.\n");
		return -1;
	}

	l= (int *) malloc(sizeof(int)*(count-1));

	if(l==NULL)
	{
		printf("Error: unknwon error in memory.\n");
		return -1;
	}
	struct words *wt=w;
		
	for (i=0;i<count-1;i++){
		wt=wt->next;
		l[i]=atof(wt->t);
		if(l[i]==0){

			printf("Error:invalid data.\n");
			return -1;
		}
	}
	//initialize the network
	if(INIT_NETWORK(l,count-1)<0){

		printf("Error: can\'t initialize the Neural Network.\n");
		free(l);
		return -1;
	}else
		printf("Network initialized succesfully.\n");
	free(l);
	return 0;
}

void doForward(struct words *w, int count){
	int i;
	double *inputs=NULL;
	double * outputs=NULL;
	struct words * wt=w;

	if(layer==NULL)
	{
		printf("Error: no network opened in memory.\n");
		return;
	}

	if(layer[0].size!=count-1){
		printf("Error: should enter %d values.\n",layer[0].size);
		return;
	}
	inputs=(double * )malloc(sizeof(double)*layer[0].size);
	if(inputs==NULL){
		printf("Error: unknown error in memory.\n");
		return;
	}

	for(i=0;i<count-1;i++){
		wt=wt->next;
		if(!isNum(wt->t))
		{
			printf("Error: invalid data.\n");
			free(inputs);
			return;
		}
		inputs[i]=atof(wt->t);
	}

	outputs=forward(inputs);

	for(i=0;i<layer[layers_count-1].size;i++)
		printf("\t%g",outputs[i]);
	printf("\n");
	
	free(inputs);

}

void doTrain(struct words * w,int count){
	
	int i;
	unsigned tr=0;
	double *input=NULL;
	double *output=NULL;
	double *target=NULL;
	struct words *wt=w;
	double cost_fn;
	struct trainSet * curSet=NULL;
	int layerSum = layer[0].size + layer[layers_count-1].size;
	int trainNum;

	if(layer==NULL)
	{
		printf("Error: no network opened in memory.\n");
		return;
	}

	if(count != layerSum + 2 && count != layerSum + 3){
		printf("Error: you should have %d inputs & %d targets.\n",layer[0].size,layer[layers_count-1].size);
		return;
	}

	input=(double *)malloc(sizeof(double)*layer[0].size);
	if(input==NULL){
		printf("Error: unknown error in memory.\n");
		return;
	}
	target=(double*)malloc(sizeof(double)*layer[layers_count].size);
	if(target==NULL){
		printf("Error: unkown error in memory.\n");
		free(input);
		return;
	}

	//getting the input values
	for(i=0;i<layer[0].size;i++)
	{
		wt=wt->next;
		if(!isNum(wt->t)){
			printf("Error: invalid data.\n");
			free(input);
			return;
		}
		input[i]=atof(wt->t);
	}

	//getting the target flag
	wt=wt->next;
	if(strcmp(wt->t,"-tar")!=0)
	{
		printf("Error: invalid data.\n");
		free(input);
		return;
	}

	//getting the target values
	for(i=0;i<layer[layers_count-1].size;i++)
	{
		wt=wt->next;
		if(!isNum(wt->t)){
			printf("Error: invalid data.\n");
			free(input);
			return;
		}
		target[i]=atof(wt->t);
	}
	
	if(count == layerSum + 2)
	{
		//add train set
		addTrainSet(input,target);
	
		do{
			curSet=trnHead;
			while(curSet!=NULL){
				output=forward(curSet->input);
				backProp(curSet->output);
				cost_fn=cost_fx(curSet->output);
				tr++;
				steps++;
				curSet=curSet->next;
			}
	
		}while(cost_fn>conv_value);
	}
	else
	{
		wt=wt->next;
		if(wt->t[0]=='r' && strlen(wt->t)>1)
		{
			trainNum=atoi(wt->t+1);
			if(trainNum)
			{
				for(i=0;i<trainNum;i++)
				{
					output=forward(input);
					backProp(output);
					tr++;
					steps++;
				}
			}
			else
			{
				printf("Error: invalid data.\n");
			}
		}
		else
		{
			printf("Error: invalid data.\n");
		}

	}


	printf("the Network was trained %u times now.\n",tr);
	printf("the Network all time trained %lu times.\n",steps);

	free(input);
	free(target);
}

void learn_rate(struct words * w, int count){
	double lR;
	if(count !=2){
		printf("Error: invalid data.\n");
		return;
	}

	w=w->next;

	if(strcmp(w->t,"-p")==0)
	{
		printf("\tlearnRate=%g\n",learnRate);
	}
	else if(isNum(w->t))
	{
		lR=atof(w->t);
		if(lR==0)
		{
			printf("Error: invalid data.\n");
			return;
		}

		learnRate=lR;

	}
	else
	{
		printf("Error: invalid data.\n");
	}

}

void convarge(struct words * w, int count){
	
	double cv;
	if(count != 2){
		printf("Error: invalid data.\n");
		return;
	}

	w=w->next;
	if(strcmp(w->t,"-p")==0)
	{
		printf("\tconvarge=%g\n",conv_value);
	}
	else if(isNum(w->t))
	{
		cv=atof(w->t);
		if(cv==0)
		{
			printf("Error: invalid data.\n");
			return;
		}

		conv_value=cv;
	}
	else
	{
		printf("Error: invalid data.\n");
	}
}

void addTrainSet(double *in, double *out){
	struct trainSet *cur=NULL;
	int i;
	if(trnHead==NULL){
		trnHead=(struct trainSet *)malloc(sizeof(struct trainSet));
		if(trnHead==NULL)
		{
			printf("Error: Unknown error in memory.\n");
			return;
		}
		cur=trnHead;
	}
	else{
		cur=trnHead;
		while(cur->next!=NULL){

			cur=cur->next;
		}
		cur->next=(struct trainSet *)malloc(sizeof(struct trainSet));
		if(cur->next==NULL)
		{
			printf("Error: Unknown error in memory.\n");
			return;
		}
		cur=cur->next;
	}

	cur->next=NULL;
	cur->input=(double *)malloc(sizeof(double)*layer[0].size);
	cur->output=(double *)malloc(sizeof(double)*layer[layers_count-1].size);
	if(cur->input==NULL || cur->output==NULL){
		printf("Error: Unkown error in memory.\n");
		return;
	}

	for(i=0;i<layer[0].size;i++)
		cur->input[i]=in[i];
	for(i=0;i<layer[layers_count-1].size;i++)
		cur->output[i]=out[i];


}

void clearTrainSet(){
	
	struct trainSet * curSet=trnHead;
	struct trainSet * tmp;
	while(curSet!=NULL){
		tmp=curSet;
		curSet=curSet->next;
		free(tmp->input);
		free(tmp->output);
		free(tmp);
	}
	trnHead=NULL;
}

int isNum(char *ch){

	int i=0,dot=0;
	if(ch[i]=='-' || ch[i]=='+')
		i++;

	for(;i<strlen(ch);i++){
		if(ch[i]=='.')
			dot++;
		else if(ch[i]<'0' || ch[i]>'9')
			return 0;
	}

	if(dot>1)
		return 0;
	return 1;
}

void showDet()
{
	if(layer==NULL)
	{
		printf("Error: NO neural network opened now.\n");
		return;
	}

	puts("Name:");
	if(filename==NULL)
	{
		puts("\t#UNTITLED#");
	}
	else
	{
		printf("\t%s\n",filename);
	}
	puts("Layers:");
	printf("\t1 input layer [%d node(s)]\n\t%d hidden layer(s)\n\t1 output layer [%d node(s)]\n"
			,layer[0].size,layers_count-2,layer[layers_count-1].size);
	puts("Learn Rate:");
	printf("\t%g\n",learnRate);
	puts("Convarge value:");
	printf("\t%g\n",conv_value);
}
