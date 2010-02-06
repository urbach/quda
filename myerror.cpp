
#include "myerror.h"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

void myerror(char v[])
{
	cout << v << endl;
	ofstream outfile("err.log");
	outfile << v << endl;
	outfile.close();
	exit(1);
}

void myerror_i(char v[], int i)
{
	cout << v << i << endl;
	ofstream outfile("err.log");
	outfile << v << i << endl;
	outfile.close();
	exit(1);
}


