#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "ann.h"
#include "neuromz.h"
#include "actFunc.h"

char QUIT=0;//for quit the command function. 


extern NEUROMZ_data * neuromz;


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
    struct words * W_head = word;
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
    //return to the start
    word = W_head;
	
	//now we star checking the line
	if(strcmp(word->t,"exit")==0)
	{
        if(neuromz)
        {
            if(neuromz->layer!=0)
                freeNet();
        }
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
        clearTrainSet();
		freeNet();
// 		if(neuromz->filename!=NULL){
// 			free(neuromz->filename);
// 			neuromz->filename=NULL;
// 		}
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
		showDet(word,count_w);
	}
	else if(strcmp(word->t,"version")==0 || strcmp(word->t,"-v")==0)
	{
		versionP();
	}
	else if(strcmp(word->t,"rmset")==0)
	{
		if(neuromz->layer==NULL)
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

	if(neuromz && neuromz->layer==NULL){
		printf("Error: no data found to save.\n");
		return;
	}

	if(count==1){
		if(neuromz->filename==0)
		{
			printf("Error: should declare a file name.\n");
			return;
		}else
		{
			name=neuromz->filename;
		}
	}
	else if(count==2)
	{
		w=w->next;
		name=w->t;
		if(neuromz->filename!=NULL)
		{
			free(neuromz->filename);
			neuromz->filename=NULL;
		}
		neuromz->filename=(char *)malloc(sizeof(char)*(strlen(name)+1));
		if(neuromz->filename==NULL){
			printf("Error: unknown error in memory.\n");
		}
		strcpy(neuromz->filename,name);
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
    
	//if(neuromz->layer != NULL || neuromz->filename != NULL){
    if(neuromz)
    {
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
		neuromz->filename=(char *)malloc(sizeof(char)*(strlen(w->t)+1));
		if(neuromz->filename==NULL){
			printf("Error: unknown error in memory.\n");
			freeNet();
			return;
		}
		strcpy(neuromz->filename,w->t);
	}

}

int insWordBefore(struct words *wd, char *ws)
{
    struct words *t, *t2;
    
    t = (struct words *) malloc(sizeof(struct words));
    if(t == NULL)
    {
        return 0;
    }
    t2 = wd->prv;
    t2->next = t;
                    
    t->prv = t2;
    t->next = wd;
    wd->prv = t;
    
    strcpy(t->t,ws);
   
    return 1;
}

struct words *deleteWord(struct words *w)
{
    struct words *t, *t2;
    
    t = w->prv;
    t2 = w->next;
    
    t->next = t2;
    if(t2 != NULL)
    {
        t2->prv = t;
    }
    free(w);
    return t2;
}

void moveStringLower(char *str,int len)
{
    int i=0;
    while(len<strlen(str))
    {
        str[i++]=str[len++];
    }
    str[i]=0;//the null terminator
}

int isDigit(char n)
{
    return (n >= '0' && n <= '9');
}

int isAlphabet(char n)
{
    return (n >= 'a' && n <= 'z') || (n >= 'A' && n <= 'Z');
}

int newNetWords(struct words * w)
{
    int opened =0;
    int l=0;
    struct words *cur;
    int i;
    char buf[32];
    enum WordType{noType,colon,openBracket,closeBracket,number,functionName};
    enum WordType type = noType;

    char *act_tmp;
    cur = w->next;//skip the command word
    while(cur)
    {
        if(cur->t[0]==':')
        {
            if(type == noType)
            {
                printf("Error: can\'t use `:` before layer.\n");
                return -1;
            }
            else if((type == number && !opened )|| type == closeBracket)
            {
                type = colon;
                
                if(strlen(cur->t)>1)
                {
                    if(insWordBefore(cur,":") == 0)
                        return -1;                    
                    moveStringLower(cur->t,1);
                }
                else
                {
                    cur = cur->next;
                }
            }
            else
            {
                printf("Error: Bad Syntax\n");
                return -1;
            }
        }
        else if(isDigit(cur->t[0]))
        {
            if(type == colon)
            {
                printf("Error: can\'t use layer after `:`.\n");
                return -1;
            }
            type = number;
            
            for (i=0;i<strlen(cur->t);i++)
            {
                if(!isDigit(cur->t[i]))
                    break;
            }
            if(i != strlen(cur->t))
            {
               
                strncpy(buf,cur->t,i);
                buf[i] = 0;
                
                if(insWordBefore(cur,buf) == 0)
                    return -1;
                moveStringLower(cur->t,i);
            }
            else
            {
                cur = cur->next;
            }
            l++;//count found numbers
        }
        else if(cur->t[0]=='[')
        {
            
            if(type == colon)
            {
                printf("Error: can\'t open bracket after `:`.\n");
                return -1;
            }
            if(type == openBracket)
            {
                printf("Error: can\'t open another bracket.\n");
                return -1;
            }
            type = openBracket;
            opened = 1;
            if(strlen(cur->t)>1)
            {

                if(insWordBefore(cur,"[") == 0)
                    return -1;
                moveStringLower(cur->t,1);
            }
            else
            {
                cur = cur->next;
            }
            
        }
        else if(cur->t[0]==']')
        {
            if(!opened)
            {
                printf("Error: can\'t close bracket without opening.\n");
                return -1;
            }
            type = closeBracket;
            opened = 0;
            if(strlen(cur->t)>1)
            {
                if(insWordBefore(cur,"]") == 0)
                    return -1;
                moveStringLower(cur->t,1);
            }
            else
            {
                cur = cur->next;
            }
        }
         else if(isAlphabet(cur->t[0]))
        {
            if(type != colon)
            {
                printf("Error: Activation function should come after `:`.\n");
                return -1;
            }
            type = functionName;
            for (i=0;i<strlen(cur->t);i++)
            {
                if(!isDigit(cur->t[i]) && !isAlphabet(cur->t[i]))
                    break;
            }
            
            if(i != strlen(cur->t))
            {
                strncpy(buf,cur->t,i);
                buf[i]=0;
                
                if(insWordBefore(cur, buf) == 0)
                    return -1;
                moveStringLower(cur->t,i);
            }
            else
            {
                act_tmp = cur->t;
                cur = cur->next;
            }
            if(getActFlag(act_tmp) == -1)
            {
                printf("Error: Not recognized Activation function.\n");
                return -1;
            }
        }
        else if(strcmp(cur->t,"") == 0)
        {
            cur = deleteWord(cur);
        }
        else
        {
            printf("Error: unknown Syntax `%s`.\n",cur->t);
            
            return -1;
        }

    }
    return l;
}

int createNet(struct words * w,int count){
	
	int i;
	int *l;
    uint16 * act;
    uint16 tmp_act;
    int opened= -1;
    int li=0;

    if(neuromz)
	{
		printf("Error: should clear current network brfore.\n");
		return -1;
	}

    count = newNetWords(w);   
    if(count == -1)
    {
        return -1;
    }
	l= (int *) calloc(count,sizeof(int));
	if(l==NULL)
	{
		printf("Error: unknwon error in memory.\n");
		return -1;
	}
	
	act = (uint16 *) malloc(sizeof(uint16)*(count));
    if(act == NULL)
    {
        free(l);
        printf("Error: unknwon error in memory.\n");
		return -1;
    }

	struct words *wt=w->next;//skip `new` keyword
    while(wt)
    {
        if(strcmp(wt->t, "[") == 0)
        {
            opened = li;
        }
        else if(isDigit(wt->t[0]))
        {
            l[li] = atoi(wt->t);
            act[li] = SIGMOID;
            li++;
        }
        else if(isAlphabet(wt->t[0]))
        {
            tmp_act = getActFlag(wt->t);
            if(opened !=-1)
            {
                for(i = opened;i<li; i++)
                {
                    act[i] = tmp_act;
                }
            }
            else
            {
                act[li - 1] = tmp_act;
            }
        }
        wt = wt->next;
    }
    
    neuromz = (NEUROMZ_data *) calloc(1,sizeof(NEUROMZ_data));
    if(neuromz == NULL)
    {
        printf("Error: Memory Failed.\n");
		free(l);
        free(act);
		return -1;
    }
    neuromz->layers_count = count;

	//initialize the network
	if(INIT_NETWORK(l,act,neuromz->layers_count)<0){

		printf("Error: can\'t initialize the Neural Network.\n");
		free(l);
        free(act);
		return -1;
	}else
		printf("Network initialized succesfully.\n");
	free(l);
    free(act);
    
	return 0;
}

void doForward(struct words *w, int count){
	int i;
	double *inputs=NULL;
	double * outputs=NULL;
	struct words * wt=w;

	if(neuromz && neuromz->layer==NULL)
	{
		printf("Error: no network opened in memory.\n");
		return;
	}

	if(neuromz->layer[0].size!=count-1){
		printf("Error: should enter %d values.\n",neuromz->layer[0].size);
		return;
	}
	inputs=(double * )malloc(sizeof(double)*neuromz->layer[0].size);
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

	for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++)
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
	int layerSum = neuromz->layer[0].size + neuromz->layer[neuromz->layers_count-1].size;
	int trainNum;

	if(neuromz == NULL && neuromz->layer==NULL)
	{
		printf("Error: no network opened in memory.\n");
		return;
	}

	if(count != layerSum + 2 && count != layerSum + 3){
		printf("Error: you should have %d inputs & %d targets.\n",neuromz->layer[0].size,neuromz->layer[neuromz->layers_count-1].size);
		return;
	}

	input=(double *)malloc(sizeof(double)*neuromz->layer[0].size);
	if(input==NULL){
		printf("Error: unknown error in memory.\n");
		return;
	}
	target=(double*)malloc(sizeof(double)*neuromz->layer[neuromz->layers_count].size);
	if(target==NULL){
		printf("Error: unkown error in memory.\n");
		free(input);
		return;
	}
	//getting the input values
	for(i=0;i<neuromz->layer[0].size;i++)
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
	for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++)
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
			curSet=neuromz->trnHead;
			while(curSet!=NULL){
				output=forward(curSet->input);
				backProp(curSet->output);
				cost_fn=cost_fx(curSet->output);
				tr++;
				neuromz->steps++;
				curSet=curSet->next;
			}
		}while(cost_fn>neuromz->conv_value);
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
					neuromz->steps++;
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
	printf("the Network all time trained %lu times.\n",neuromz->steps);

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
    if(neuromz == NULL)
    {
        printf("Error: no network opened in memory.\n");
		return;
    }

	if(strcmp(w->t,"-p")==0)
	{
		printf("\tlearnRate=%g\n",neuromz->learnRate);
	}
	else if(isNum(w->t))
	{
		lR=atof(w->t);
		if(lR==0)
		{
			printf("Error: invalid data.\n");
			return;
		}

		neuromz->learnRate=lR;

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
	
    if(neuromz == NULL)
    {
        printf("Error: no network opened in memory.\n");
		return;
    }
	w=w->next;
	if(strcmp(w->t,"-p")==0)
	{
		printf("\tconvarge=%g\n",neuromz->conv_value);
	}
	else if(isNum(w->t))
	{
		cv=atof(w->t);
		if(cv==0)
		{
			printf("Error: invalid data.\n");
			return;
		}

		neuromz->conv_value=cv;
	}
	else
	{
		printf("Error: invalid data.\n");
	}
}

void addTrainSet(double *in, double *out){
	struct trainSet *cur=NULL;
	int i;
    
    if(neuromz == NULL)
    {
        printf("Error: no network opened in memory.\n");
		return;
    }
    
	if(neuromz->trnHead==NULL){
		neuromz->trnHead=(struct trainSet *)malloc(sizeof(struct trainSet));
		if(neuromz->trnHead==NULL)
		{
			printf("Error: Unknown error in memory.\n");
			return;
		}
		cur=neuromz->trnHead;
	}
	else{
		cur=neuromz->trnHead;
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
	cur->input=(double *)malloc(sizeof(double)*neuromz->layer[0].size);
	cur->output=(double *)malloc(sizeof(double)*neuromz->layer[neuromz->layers_count-1].size);
	if(cur->input==NULL || cur->output==NULL){
		printf("Error: Unkown error in memory.\n");
		return;
	}

	for(i=0;i<neuromz->layer[0].size;i++)
		cur->input[i]=in[i];
	for(i=0;i<neuromz->layer[neuromz->layers_count-1].size;i++)
		cur->output[i]=out[i];
	
}

void clearTrainSet(){
    if(neuromz == NULL)
    {
        printf("Error: no network opened in memory.\n");
		return;
    }
	struct trainSet * curSet=neuromz->trnHead;
	struct trainSet * tmp;
	while(curSet!=NULL){
		tmp=curSet;
		curSet=curSet->next;
		free(tmp->input);
		free(tmp->output);
		free(tmp);
	}
	neuromz->trnHead=NULL;
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

void showDet(struct words * w,int count)
{
    int i;
    struct words * tmp = w->next;
    int all = 0;
    if(neuromz == NULL)
    {
        printf("Error: no network opened in memory.\n");
		return;
    }
	if(neuromz->layer==NULL)//should be deleted
	{
		printf("Error: NO neural network opened now.\n");
		return;
	}
	
	if(count>1)
    {
        while(tmp)
        {
            if(strcmp(tmp->t, "all") == 0)
            {
                all = 1;
            }
            else
            {
                if(strlen(tmp->t)>0)
                {
                    printf("Error: Syntax Error.\n");
                    return;
                }
            }
            
            tmp = tmp->next;
        }
    }
	

	puts("Name:");
	if(neuromz->filename==NULL)
	{
		puts("\t#UNTITLED#");
	}
	else
	{
		printf("\t%s\n",neuromz->filename);
	}
	puts("Layers:");
	printf("\t1 input layer [%d node(s)][Activation `%s`]\n",neuromz->layer[0].size,getActName(neuromz->layer[0].ACT_FX));
	printf("\t%d hidden layer(s)\n",neuromz->layers_count-2);
    if(all == 1)
    {
        for(i=1; i<neuromz->layers_count-1; i++)
        {
            printf("\t\tlayer%d : [%d node(s)][Activation `%s`]\n",i, neuromz->layer[i].size, 
                   getActName(neuromz->layer[i].ACT_FX));
        }
    }
    printf("\t1 output layer [%d node(s)][Activation `%s`]\n",neuromz->layer[neuromz->layers_count-1].size, getActName(neuromz->layer[neuromz->layers_count-1].ACT_FX));
    

	puts("Learn Rate:");
	printf("\t%g\n",neuromz->learnRate);
	puts("Convarge value:");
	printf("\t%g\n",neuromz->conv_value);
}
