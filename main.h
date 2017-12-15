#ifndef _MAINP_H_
#define _MAINP_H_

#include "neuromz.h"

void command();
void testArg(int argc,char * argv[]);

void getCMD(char*text , int limit);
void getStarter();

void versionP();
void printHelp(struct words *w, int count);

#endif
