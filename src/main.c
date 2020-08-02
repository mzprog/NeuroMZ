#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "neuromz.h"
#include "ann.h"


extern char QUIT;//for quit the command function. 

// extern char * filename;
// extern struct layers * layer;
// extern uint16 layers_count;

extern NEUROMZ_data *neuromz;

int main(int argc,char * argv[]){

	if(argc<2)
		command();
	else
		testArg(argc,argv);

	return 0;
}


void command(){
	char line[256];
	printf("NeuroMZ 2017\nTo more help type \"-h\" or \"help\".\n");

	while(!QUIT){
		//get starter.
		getStarter();
		getCMD(line,256);//get the command
		checkLine(line);
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
			if(neuromz->filename!=NULL)
			{
				free(neuromz->filename);
				neuromz->filename=NULL;
			}
			neuromz->filename=(char *) malloc(sizeof(char) * (strlen(argv[i])+1));
			strcpy(neuromz->filename,argv[++i]);

			SAVE=1;
		}
		else if(strcmp(argv[i],"-test")==0)
		{
			if(neuromz->layer==NULL)
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
			if(neuromz->layer[0].size!=j-i)
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
			if(neuromz->layer==NULL)
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
		
			if(neuromz->layer[0].size!=j-i)
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
			if(neuromz->layer[neuromz->layers_count-1].size!=j-i)
			{
				printf("Error: Invalid output layer count.\n");
				ERROR=1;
				break;
			}
			if(j<argc && argv[j][0]=='r' && isNum((argv[j]+1)))
			{
				newNode(argv[j++],word);
				doTrain(word,3+neuromz->layer[0].size+neuromz->layer[neuromz->layers_count-1].size);
			}
			else
			{
				doTrain(word,2+neuromz->layer[0].size+neuromz->layer[neuromz->layers_count-1].size);
			}
			clearWords(word);
			i=j-1;
			SAVE=1;
		}
		else if(strcmp(argv[i],"-lr")==0)
		{
			if(neuromz->layer==NULL)
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
			if(neuromz->layer==NULL)
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
			if(neuromz->layer==NULL)
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
			if(neuromz->filename==NULL)
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
	if(neuromz->layer!=NULL)
	{
		freeNet();
		clearTrainSet();
		if(neuromz->filename!=NULL){
			free(neuromz->filename);
			neuromz->filename=NULL;
		}
	//some fix the filename args or malloc to free it
	}

}


void getCMD(char * text, int limit){
	int len;
	fgets(text,limit,stdin);
	len=strlen(text);
	if(len>0 && text[len-1]=='\n')
		text[len-1]=0;
}

void getStarter(){

	if(neuromz == NULL)
    {
        printf("NO-NETWORK");
    }
    else if(neuromz->filename == NULL)
    {
        printf("UNTITLED");
    }
	else
		printf("%s",neuromz->filename);

	printf(">>>");
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
		puts("  version\tDisplay the version of NeuroMZ");
		puts("  exit\t\tExit the Neuromz command line.");
		puts("  help\t\tPrint help.");

		puts("\n");
		
		puts("For Passing By Args:\n");
		puts("  -new\t\tCreate new neural network.");
		puts("  -file\t\tLoad neural network file.");
                puts("  -train\tTrain the neural network.");
                puts("  -tar\t\tTagets value fo the train command.");
                puts("  -test\t\tTest the network, And do forward.");
                puts("  -lr\t\tPrint or Edit the learn rate value.");
                puts("  -conv\t\tPrint or Edit the convarge value.");
                puts("  -rmset\tRemove the trainning set");
                puts("  -show\t\tShow the neural network details.");
                puts("  -version\tDisplay the version of NeuroMZ");
                puts("  --help\tPrint help.");
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
