#ifndef _TOOLS_H_
#define _TOOLS_H_

struct words{
	char t[32];
	struct words * prv;
	struct words * next;
};

void clearWords(struct words *w);
void checkLine(char * line);
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

#endif
