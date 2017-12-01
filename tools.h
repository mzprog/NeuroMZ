#ifndef _TOOLS_H_
#define _TOOLS_H_

char * filename=0;
char line[256];
double conv_value=0.005;
char QUIT=0;
struct words{
	char t[32];
	struct words * prv;
	struct words * next;
};

struct trainSet{

	double * input;
	double * output;
	struct trainSet * next;
};

struct trainSet * trnHead=NULL;

void command();
void testArg(int argc,char * argv[]);
void clearWords(struct words *w);
void getCMD(char*text , int limit);
void getStarter();
void checkLine();
void newNode(char * txt,struct words * w);
void printError();
//the command functions 
void saveFile(struct words * w, int count);
void loadFile(struct words * w, int count);
int createNet(struct words * w,int count);
void doForward(struct words* w,int count);
void doTrain(struct words *w,int count);
void learn_rate(struct words * w, int count);
void convarge(struct words * w,int count);
void addTrainSet(double *in,double * out );//should convert to int
void clearTrainSet();
int isNum(char * ch);
void showDet();
void versionP();
void printHelp(struct words *w, int count);
#endif
