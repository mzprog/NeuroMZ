#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"
#include "neuromz.h"
#include "config.h"


extern struct trainSet * trnHead;
extern double *  temp;//we should allocate as the bigest layer neurons.
extern char * filename;
extern double conv_value;
extern struct layers * layer;
extern uint16 layers_count;
extern unsigned long steps;
extern double * output_val;
extern double learnRate;//we can change it later by the main function



int main(int argc,char * argv[]){

	if(argc<2)
		command();
	else
		testArg(argc,argv);

	return 0;
}


void command(){
	printf("NeuroMZ 2017\nTo more help type \"-h\" or \"help\".\n");
	while(!QUIT){
		//get starter.
		getStarter();
		getCMD(line,256);//get the command
		checkLine();
	}

}

void testArg(int argc, char *argv[]){

	int i,j;
	int SAVE=0,ERROR=0;//bollean variable
	struct words * word=(struct words *)malloc(sizeof(struct words));
	word->prv=NULL;
	word->next=NULL;


	for(i=1;i<argc;i++){
		if(strcmp(argv[i],"-new")==0)
		{
			newNode(argv[i],word);
			j=i+1;
			while(j<argc && isNum(argv[j]))
			{
				newNode(argv[j],word);
				j++;
			}
			
			createNet(word,j-i);
			i=j-1;
			clearWords(word);
			SAVE=1;
		}
		else if(strcmp(argv[i],"-file")==0)
		{
			if(i+1>=argc)
			{
				printf("Error: no value for the file name.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i++],word);
			newNode(argv[i],word);
			loadFile(word,2);
			clearWords(word);
		}
		else if(strcmp(argv[i],"-name")==0)
		{
			if(i+1>=argc)
			{
				printf("Error: no value for the name.\n");
				ERROR=1;
				break;
			}
			if(filename!=NULL)
			{
				free(filename);
				filename=NULL;
			}
			filename=(char *) malloc(sizeof(char) * (strlen(argv[i])+1));
			strcpy(filename,argv[++i]);

			SAVE=1;
		}
		else if(strcmp(argv[i],"-test")==0)
		{
			if(layer==NULL)
			{
				printf("Error: Should open neural network first.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i++],word);
			j=i;
			while(j<argc && isNum(argv[j]))
			{
				newNode(argv[j++],word);
			}
			if(layer[0].size!=j-i)
			{
				printf("Error: Invalid input layer count.\n");
				ERROR=1;
				break;
			}
			doForward(word,j-i+1);
			i=j-1;
			clearWords(word);
		}
		else if(strcmp(argv[i],"-train")==0)
		{
			if(layer==NULL)
			{
				printf("Error: Should open neural network first.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i++],word);
			j=i;
			while(j<argc && isNum(argv[j]))
			{
				newNode(argv[j++],word);
			}
		
			if(layer[0].size!=j-i)
			{
				printf("Error: Invalid input layer count.\n");
				ERROR=1;
				break;
			}
			
			if(j>=argc)
			{
				printf("Error: not enough data to do the train.\n");
				ERROR=1;
				break;
			}
	
			if(strcmp(argv[j],"-tar")!=0)
			{
				printf("Error: target flag [-tar] is not found.\n");
				ERROR=1;
				break;
			}

			newNode(argv[j++],word);
			i=j;
			while(j<argc && isNum(argv[j]))
			{
				newNode(argv[j++],word);
			}
			if(layer[layers_count-1].size!=j-i)
			{
				printf("Error: Invalid output layer count.\n");
				ERROR=1;
				break;
			}
			if(j<argc && argv[j][0]=='r' && isNum((argv[j]+1)))
			{
				newNode(argv[j++],word);
				doTrain(word,3+layer[0].size+layer[layers_count-1].size);
			}
			else
			{
				doTrain(word,2+layer[0].size+layer[layers_count-1].size);
			}
			clearWords(word);
			i=j-1;
			SAVE=1;
		}
		else if(strcmp(argv[i],"-lr")==0)
		{
			if(layer==NULL)
			{
				printf("Error: should open neural network first.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i++],word);
			
			if(i>=argc)
			{
				printf("Error: no value for learn rate.\n");
				ERROR=1;
				break;
			}

			if(!isNum(argv[i]))
			{
				printf("Error: invalid learn rate value.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i],word);
			learn_rate(word,2);
			clearWords(word);
			SAVE=1;
		}
		else if(strcmp(argv[i],"-conv")==0)
		{
			if(layer==NULL)
			{
				printf("Error: should open neural network first.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i++],word);

			if(i>=argc)
			{
				printf("Error: no value for convargent.\n");
				ERROR=1;
				break;
			}

			if(!isNum(argv[i]))
			{
				printf("Error: invalid convargent value.\n");
				ERROR=1;
				break;
			}
			newNode(argv[i],word);
			convarge(word,2);
			clearWords(word);
			SAVE=1;
		}
		else if(strcmp(argv[i],"-show")==0)
		{
			showDet();
		}
		else if(strcmp(argv[i],"-v")==0)
		{
			versionP();
		}
		else if(strcmp(argv[i],"-h")==0 || strcmp(argv[i],"--help")==0)
		{
			if(argc==2 && i==1)
			{
				newNode(argv[i],word);
				printHelp(word,1);
				clearWords(word);
			}
			else if(argc==3 && i==1)
			{
				newNode(argv[i++],word);
				newNode(argv[i],word);
				printHelp(word,2);
				clearWords(word);
			}
			else 
			{
				printf("Error: to many data.\n");
			}
		}
		else if(strcmp(argv[i],"-rmset")==0)
		{
			if(layer==NULL)
			{
				printf("Error: choose a neural network to clear its train set.\n");
				ERROR=1;
				break;
			}
			clearTrainSet();
		}
		else 
		{
			ERROR=1;
			printf("Error: invalid args [%s].\n",argv[i]);
			break;
		}
	}
	//after checking the arguement
	if(SAVE)
	{
		if(ERROR)
		{
			printf("Can\'t save the changes due to some errors.\n");
		}
		else
		{
			if(filename==NULL)
			{
				printf("Error: Can\'t save without file name.\n");
			}
			else
			{
				saveFile(NULL,1);
			}
		}
	}
	//if we have some words list not cleared because of the break
	clearWords(word);
	//clear last word list 
	free(word);
	if(layer!=NULL)
	{
		freeNet();
		clearTrainSet();
		if(filename!=NULL){
			free(filename);
			filename=NULL;
		}
	//some fix the filename args or malloc to free it
	}

}

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

void getCMD(char * text, int limit){
	int len;
	fgets(text,limit,stdin);
	len=strlen(text);
	if(len>0 && text[len-1]=='\n')
		text[len-1]=0;
}

void getStarter(){

	if(filename==0)
		if(layer==0)
			printf("NO-NETWORK");
		else
			printf("UNTITLED");
	else
		printf("%s",filename);

	printf(">>>");
}

void checkLine(){

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
	printf("\t1 input layer [%d nodes]\n\t%d hiden layers\n\t1 output layer [%d nodes]\n"
			,layer[0].size,layers_count-2,layer[layers_count-1].size);
	puts("Learn Rate:");
	printf("\t%g\n",learnRate);
	puts("Convarge value:");
	printf("\t%g\n",conv_value);
}

void versionP()
{
	puts("NeuroMZ 1.0");
}

void printHelp(struct words *w, int count)
{
	if(count == 1)
	{	
		puts("For Neuromz Command line:\n");
		puts("  new\t\tCreate new neural network.");
		puts("  load\t\tLoad neural network file.");
		puts("  save\t\tSave neural network file.");
		puts("  clear\t\tClear current neural network from the memory.");
		puts("  train\t\tTrain the neural network.");
		puts("  -tar\t\tTagets value fo the train command.");
		puts("  test\t\tTest the network, And do forward.");
		puts("  learnrate\tPrint or Edit the learn rate value.");
		puts("  conv\t\tPrint or Edit the convarge value.");
		puts("  rmset\t\tRemove the trainning set");
		puts("  show\t\tShow the neural network details.");
		puts("  version\t\tDisplay the version of NeuroMZ");
		puts("  exit\t\tExit the Neuromz command line.");
		puts("  help\t\tPrint help.");

		puts("\n");
		
		puts("For Passing By Args:\n");
		puts("  -new\t\tCreate new neural network.");
		puts("  -file\t\tLoad neural network file.");
                puts("  -train\t\tTrain the neural network.");
                puts("  -tar\t\tTagets value fo the train command.");
                puts("  -test\t\tTest the network, And do forward.");
                puts("  -lr\tPrint or Edit the learn rate value.");
                puts("  -conv\t\tPrint or Edit the convarge value.");
                puts("  -rmset\t\tRemove the trainning set");
                puts("  -show\t\tShow the neural network details.");
                puts("  -version\t\tDisplay the version of NeuroMZ");
                puts("  --help\t\tPrint help.");
	}
	else if(count == 2)
	{
		w=w->next;
		if(strcmp(w->t,"new")==0)
		{
			puts("new: to create new neural network by declare the size of each layer.");
			puts("\tnew [l1 l2 l3 ... ln]");

		}
		else if(strcmp(w->t,"load")==0)
		{
			puts("load: to load a saved file of neural network");
			puts("\tload [filename]");
		}
		else if(strcmp(w->t,"save")==0)
		{
			puts("save: to save neural network to a file");
			puts("if the file name exist you can just type save or you can give another file name");
			puts("\tsave [filename]");
			puts("or just type:");
			puts("\tsave");
		}
		else if(strcmp(w->t,"clear")==0)
		{
			puts("clear: to clear or close the neural network from the memory to load or create another one");
			puts("\tclear");
		}
		else if(strcmp(w->t,"train")==0)
		{
			puts("train: to train the neural network");
			puts("\ttrain [in1 in2 .. inn] -tar [out1 out 2 .. outn] {rn}");
			puts("-tar: the target flag to seperate the inputs from targets");
			puts("rn: the rn optional to use while it\'s repeat trainning n times");
			puts("\tBy using {rn} you can train just your trainnig data for number of time");
			puts("\twithout {rn} by default you will use trainning set.");
			puts("example:");
			puts("\ttrain 1 4 -tar 3 r200  ##with rn");
			puts("\ttrain 1 4 -tar 3");
		}
		else if(strcmp(w->t,"-tar")==0)
		{
			puts("-tar : is the target values flag used with train command");
		}
		else if(strcmp(w->t,"test")==0)
		{
			puts("test: to test and do forward with a neural network");
			puts("and the result of all outputs will print on the terminal");
			puts("\ttest [in1 in2 .. inn]");
		}
		else if(strcmp(w->t,"learnrate")==0)
		{
			puts("learnrate: to change the learnrate");
			puts("\tlearnrate [value]");
		}
		else if(strcmp(w->t,"conv")==0)
		{
			puts("conv: to change the convarge of the cost function for training");
			puts("\tconv [value]");
		}
		else if(strcmp(w->t,"rmset")==0)
		{
			puts("rmset: to remove the train set lists.");
			puts("\tthe train set is to train many data at the same time.");
			puts("\tex: if you have 3 sets t1 t2 t3 will train:");
			puts("\t\t t1 t2 t3 t1 t2 t3 t1 t2 t3 .... until cost function convarge");
			puts("\tit\'s useful as a basics of trainning for the neural network");
		}
		else if(strcmp(w->t,"show")==0)
		{
			puts("show: to show details of the neural network file.");
			puts("showing the filename, layers cout, number of inputs node and output");
		}
		else if(strcmp(w->t,"version")==0)
		{
			puts("version: to print and display version number of NeuroMZ.");
		}
		else if(strcmp(w->t,"exit")==0)
		{
			puts("exit: to exit NeuroMZ");
			puts("\texit");
		}
		else if(strcmp(w->t,"help")==0)
		{
			puts("help: to print help");
			puts("\thelp");
			puts("to print help in general");
			puts("\thelp [command]");
			puts("to print help for a one command");
		}
		else
		{
			printf("Error: command \"%s\" is not found\n",w->t);
		}
	}
	else
	{
		puts("Error: invalid data");
	}

	puts("If there exits any bug or error please contact us: mz32programmer@gmail.com");
}
