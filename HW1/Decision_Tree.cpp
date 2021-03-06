#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#define RAND_SIZE 1000000

using namespace std;

class Flower{
public:
	float sepal_len;
	float sepal_wid;
	float petal_len;
	float petal_wid;
	string type;
	int randNum;
};

class Boundary{
public:
	float threshold;
	float rem;
	float IG;
};

class Classification{
public:
	string feature;
	float Threshold;
	float Ig;
	bool same;
};

class Answer{
public:
	float Se_precision;
	float Color_precision;
	float Ca_precision;
	float Se_recall;
	float Color_recall;
	float Ca_recall;
	float Accuracy;
};

struct DTree{
	Classification ClassifyFeature;
	bool leave;
	string flowerType;
	struct DTree *left;
	struct DTree *right;
};
typedef struct DTree treenode;
typedef treenode *dtree;



Classification ComputeIG(vector<Flower> &info); // compute four descriptive features' IG

void BuildTree(dtree root, vector<Flower> &f); // insert two nodes once a time

Answer TestData(dtree root, vector<Flower> &f); // test data

bool compare(const Flower &f1, const Flower &f2){
	return (f1.randNum < f2.randNum);
}

bool compare1(const Flower &f1, const Flower &f2){
	return (f1.sepal_len < f2.sepal_len);
}

bool compare2(const Flower &f1, const Flower &f2){
	return (f1.sepal_wid < f2.sepal_wid);
}

bool compare3(const Flower &f1, const Flower &f2){
	return (f1.petal_len < f2.petal_len);
}

bool compare4(const Flower &f1, const Flower &f2){
	return (f1.petal_wid < f2.petal_wid);
}

bool compare5(const Boundary &b1, const Boundary &b2){
	return (b1.IG > b2.IG);
}

int main(){
	fstream infile;
	infile.open("iris.txt",ios::in);
	string s1;
	vector<string> Vs1;
	
	//-------------------------read file-----------------------------
	if(infile.is_open()){
		while(!infile.eof()){
			getline(infile, s1);
			Vs1.push_back(s1);
		}
	}

	int flowerInfoIndex = 0;
	int flowerNum = Vs1.size();
	vector<Flower> flower; // all flower data in this
	string ts;
	ts.clear();

	flower.resize(flowerNum);

	for(int i = 0; i < flowerNum; i++){
		for(int j = 0; j < Vs1[i].size(); j++){
			if(Vs1[i][j] != ','){
				ts = ts + Vs1[i][j];
				if(j == Vs1[i].size()-1){
					flower[i].type = ts;
					ts.clear();
				}
			}
			else{
				if(flowerInfoIndex == 0){
					flower[i].sepal_len = strtof(ts.c_str(), 0);
					ts.clear();
					flowerInfoIndex++;
				}
				else if(flowerInfoIndex == 1){
					flower[i].sepal_wid = strtof(ts.c_str(), 0);
					ts.clear();
					flowerInfoIndex++;
				}
				else if(flowerInfoIndex == 2){
					flower[i].petal_len = strtof(ts.c_str(), 0);
					ts.clear();
					flowerInfoIndex++;
				}
				else if(flowerInfoIndex == 3){
					flower[i].petal_wid = strtof(ts.c_str(), 0);
					ts.clear();
					flowerInfoIndex = 0;
				}
			}
		}
	}

	flower.pop_back();
	//-----------------------------------------------------------------

	/*for(int i = 0; i < flower.size(); i++){
		cout << flower[i].sepal_len << " " << flower[i].sepal_wid << " " << flower[i].petal_len << " " << flower[i].petal_wid << " " << flower[i].type << endl;
	}*/

	//---------------------------random shuffle------------------------
	srand(time(NULL));
	for(int i = 0; i < flower.size(); i ++){ 
		flower[i].randNum = (rand() % RAND_SIZE);
	}
	sort(flower.begin(), flower.end(), compare);
	//-----------------------------------------------------------------

	//------------------K-fold cross validation (K = 5)----------------
	int trainNum = (flower.size()) * 4 / 5; // 90
	int testNum = (flower.size()) / 5; //30
	vector<Flower> TrainFlower;
	vector<Flower> TestFlower;
	TrainFlower.resize(trainNum);
	TestFlower.resize(testNum);
	Answer final[5];

	for(int i = 0; i < 5; i++){
		if(i == 0){ // first validation step
			for(int j = 0; j < TrainFlower.size(); j++){ // traing data
				TrainFlower[j] = flower[j];
			}

			// building decision tree
			// create Root first
			dtree Root = NULL;
			dtree tnode = new treenode;
			tnode->ClassifyFeature = ComputeIG(TrainFlower); // ClassifyFeature
			srand(time(NULL));
			for(int i = 0; i < TrainFlower.size(); i ++){ 
				TrainFlower[i].randNum = (rand() % RAND_SIZE);
			}
			sort(TrainFlower.begin(), TrainFlower.end(), compare);
			tnode->leave = false; // leave
			Root = tnode;
			BuildTree(Root, TrainFlower);

			for(int j = 0; j < TestFlower.size(); j++){ // test data
				TestFlower[j] = flower[j+trainNum];
			}

			final[0] = TestData(Root, TestFlower);
		}
		else if(i == 1){
			for(int j = 0; j < 90; j++){
				TrainFlower[j] = flower[j];
			}
			for(int j = 120; j < 150; j++){
				TrainFlower[j-30] = flower[j];
			}

			dtree Root = NULL;
			dtree tnode = new treenode;
			tnode->ClassifyFeature = ComputeIG(TrainFlower); 
			srand(time(NULL));
			for(int i = 0; i < TrainFlower.size(); i ++){ 
				TrainFlower[i].randNum = (rand() % RAND_SIZE);
			}
			sort(TrainFlower.begin(), TrainFlower.end(), compare);
			tnode->leave = false;
			Root = tnode;
			BuildTree(Root, TrainFlower);

			for(int j = 90; j < 120; j++){
				TestFlower[j-90] = flower[j];
			}

			final[1] = TestData(Root, TestFlower);
		}
		else if(i == 2){
			for(int j = 0; j < 60; j++){
				TrainFlower[j] = flower[j];
			}
			for(int j = 90; j < 150; j++){
				TrainFlower[j-30] = flower[j];
			}

			dtree Root = NULL;
			dtree tnode = new treenode;
			tnode->ClassifyFeature = ComputeIG(TrainFlower); 
			srand(time(NULL));
			for(int i = 0; i < TrainFlower.size(); i ++){ 
				TrainFlower[i].randNum = (rand() % RAND_SIZE);
			}
			sort(TrainFlower.begin(), TrainFlower.end(), compare);
			tnode->leave = false;
			Root = tnode;
			BuildTree(Root, TrainFlower);

			for(int j = 60; j < 90; j++){
				TestFlower[j-60] = flower[j];
			}

			final[2] = TestData(Root, TestFlower);
		}
		else if(i == 3){
			for(int j = 0; j < 30; j++){
				TrainFlower[j] = flower[j];
			}
			for(int j = 60; j < 150; j++){
				TrainFlower[j-30] = flower[j];
			}

			dtree Root = NULL;
			dtree tnode = new treenode;
			tnode->ClassifyFeature = ComputeIG(TrainFlower); 
			srand(time(NULL));
			for(int i = 0; i < TrainFlower.size(); i ++){ 
				TrainFlower[i].randNum = (rand() % RAND_SIZE);
			}
			sort(TrainFlower.begin(), TrainFlower.end(), compare);
			tnode->leave = false;
			Root = tnode;
			BuildTree(Root, TrainFlower);

			for(int j = 30; j < 60; j++){
				TestFlower[j-30] = flower[j];
			}
			final[3] = TestData(Root, TestFlower);
		}
		else if(i == 4){
			for(int j = 30; j < 150; j++){
				TrainFlower[j-30] = flower[j];
			}

			dtree Root = NULL;
			dtree tnode = new treenode;
			tnode->ClassifyFeature = ComputeIG(TrainFlower); 
			srand(time(NULL));
			for(int i = 0; i < TrainFlower.size(); i ++){ 
				TrainFlower[i].randNum = (rand() % RAND_SIZE);
			}
			sort(TrainFlower.begin(), TrainFlower.end(), compare);
			tnode->leave = false;
			Root = tnode;
			BuildTree(Root, TrainFlower);

			for(int j = 0; j < 30; j++){
				TestFlower[j] = flower[j];
			}
			final[4] = TestData(Root, TestFlower);
		}
	}
	//-----------------------------------------------------------------
	// compute total Accuracy , precision and recall
	Answer a;
	float sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Se_precision;
	}
	a.Se_precision = sum / 5;
	sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Color_precision;
	}
	a.Color_precision = sum / 5;
	sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Ca_precision;
	}
	a.Ca_precision = sum / 5;
	sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Se_recall;
	}
	a.Se_recall = sum / 5;
	sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Color_recall;
	}
	a.Color_recall = sum / 5;
	sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Ca_recall;
	}
	a.Ca_recall = sum / 5;
	sum = 0;
	for(int i = 0; i < 5; i++){
		sum += final[i].Accuracy;
	}
	a.Accuracy = sum / 5;
	sum = 0;

	// print the output
	/*cout << a.Accuracy << endl;
	cout << a.Se_precision << " " << a.Se_recall << endl;
	cout << a.Color_precision << " " << a.Color_recall << endl;
	cout << a.Ca_precision << " " << a.Ca_recall << endl;*/
	printf("%.3f\n", a.Accuracy);
	printf("%.3f %.3f\n", a.Se_precision, a.Se_recall);
	printf("%.3f %.3f\n", a.Color_precision, a.Color_recall);
	printf("%.3f %.3f\n", a.Ca_precision, a.Ca_recall);

	return 0;
}

//-----------------------ComputeIG function----------------------------
Classification ComputeIG(vector<Flower> &info){

	if(info.size() == 0){
		Classification cla;
		cla.feature = "serious_error";
		return cla;
	}

	Boundary Maxb[4];
	// compute total entropy
	int c1 = 0; // Iris-setosa
	int c2 = 0; // Iris-versicolor
	int c3 = 0; // Iris-virginica
	float proC1, proC2, proC3;
	for(int j = 0; j < info.size(); j++){
		if(info[j].type == "Iris-setosa")
			c1++;
		else if(info[j].type == "Iris-versicolor")
			c2++;
		else if(info[j].type == "Iris-virginica")
			c3++;
	}
	proC1 = (float)c1 / (float)(info.size());
	proC2 = (float)c2 / (float)(info.size());
	proC3 = (float)c3 / (float)(info.size());
	float part1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
	float part2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
	float part3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
	float H = -(part1 + part2 + part3); // H is total entropy

	//----compute sepal_len's IG----
	// according to feature's number to sort in ascending
	sort(info.begin(), info.end(), compare1);
	// set boundary between different level
	vector<Boundary> bound1;
	Boundary tempBound;
	for(int i = 0; i < info.size()-1; i++){
		if(info[i].type != info[i+1].type){
			// compute threshold between each different level
			float thre = (info[i].sepal_len + info[i+1].sepal_len) / 2;
			tempBound.threshold = thre;
			// compute rem of threshold
			float tempRem;
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = 0; j <= i; j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(i+1));
			proC2 = ((float)c2 / (float)(i+1));
			proC3 = ((float)c3 / (float)(i+1));
			float p1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float p2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float p3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float lw = (float)(i+1)/(float)(info.size()) * (-(p1+p2+p3));
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = i+1; j < info.size(); j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(info.size()-i-1));
			proC2 = ((float)c2 / (float)(info.size()-i-1));
			proC3 = ((float)c3 / (float)(info.size()-i-1));
			float P1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float P2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float P3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float hi = (float)(info.size()-i-1)/(float)(info.size()) * (-(P1+P2+P3));
			tempRem = lw + hi;
			tempBound.rem = tempRem;
			// compute IG
			float ig = H - tempRem;
			tempBound.IG = ig;

			bound1.push_back(tempBound);
		}
	}
	if(bound1.size() != 0){
		sort(bound1.begin(), bound1.end(), compare5);
		Maxb[0] = bound1[0];
	}
	else{
		tempBound.rem = 0;
		tempBound.IG = H - tempBound.rem;
		tempBound.threshold = 0;
		bound1.push_back(tempBound);
		Maxb[0] = bound1[0];
	}
	//------------------------------

	//----compute sepal_wid's IG----
	sort(info.begin(), info.end(), compare2);
	vector<Boundary> bound2;
	for(int i = 0; i < info.size()-1; i++){
		if(info[i].type != info[i+1].type){
			// compute threshold between each different level
			float thre = (info[i].sepal_wid + info[i+1].sepal_wid) / 2;
			tempBound.threshold = thre;
			// compute rem of threshold
			float tempRem;
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = 0; j <= i; j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(i+1));
			proC2 = ((float)c2 / (float)(i+1));
			proC3 = ((float)c3 / (float)(i+1));
			float p1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float p2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float p3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float lw = (float)(i+1)/(float)(info.size()) * (-(p1+p2+p3));
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = i+1; j < info.size(); j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(info.size()-i-1));
			proC2 = ((float)c2 / (float)(info.size()-i-1));
			proC3 = ((float)c3 / (float)(info.size()-i-1));
			float P1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float P2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float P3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float hi = (float)(info.size()-i-1)/(float)(info.size()) * (-(P1+P2+P3));
			tempRem = lw + hi;
			tempBound.rem = tempRem;
			// compute IG
			float ig = H - tempRem;
			tempBound.IG = ig;

			bound2.push_back(tempBound);
		}
	}
	if(bound2.size() != 0){
		sort(bound2.begin(), bound2.end(), compare5);
		Maxb[1] = bound2[0];
	}
	else{
		tempBound.rem = 0;
		tempBound.IG = H - tempBound.rem;
		tempBound.threshold = 0;
		bound2.push_back(tempBound);
		Maxb[1] = bound2[0];
	}
	//------------------------------

	//----compute petal_len's IG----
	sort(info.begin(), info.end(), compare3);
	vector<Boundary> bound3;
	for(int i = 0; i < info.size()-1; i++){
		if(info[i].type != info[i+1].type){
			// compute threshold between each different level
			float thre = (info[i].petal_len + info[i+1].petal_len) / 2;
			tempBound.threshold = thre;
			// compute rem of threshold
			float tempRem;
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = 0; j <= i; j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(i+1));
			proC2 = ((float)c2 / (float)(i+1));
			proC3 = ((float)c3 / (float)(i+1));
			float p1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float p2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float p3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float lw = (float)(i+1)/(float)(info.size()) * (-(p1+p2+p3));
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = i+1; j < info.size(); j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(info.size()-i-1));
			proC2 = ((float)c2 / (float)(info.size()-i-1));
			proC3 = ((float)c3 / (float)(info.size()-i-1));
			float P1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float P2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float P3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float hi = (float)(info.size()-i-1)/(float)(info.size()) * (-(P1+P2+P3));
			tempRem = lw + hi;
			tempBound.rem = tempRem;
			// compute IG
			float ig = H - tempRem;
			tempBound.IG = ig;

			bound3.push_back(tempBound);
		}
	}
	if(bound3.size() != 0){
		sort(bound3.begin(), bound3.end(), compare5);
		Maxb[2] = bound3[0];
	}
	else{
		tempBound.rem = 0;
		tempBound.IG = H - tempBound.rem;
		tempBound.threshold = 0;
		bound3.push_back(tempBound);
		Maxb[2] = bound3[0];
	}
	//------------------------------

	//----compute petal_wid's IG----
	sort(info.begin(), info.end(), compare4);
	vector<Boundary> bound4;
	for(int i = 0; i < info.size()-1; i++){
		if(info[i].type != info[i+1].type){
			// compute threshold between each different level
			float thre = (info[i].petal_wid + info[i+1].petal_wid) / 2;
			tempBound.threshold = thre;
			// compute rem of threshold
			float tempRem;
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = 0; j <= i; j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(i+1));
			proC2 = ((float)c2 / (float)(i+1));
			proC3 = ((float)c3 / (float)(i+1));
			float p1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float p2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float p3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float lw = (float)(i+1)/(float)(info.size()) * (-(p1+p2+p3));
			c1 = 0;
			c2 = 0;
			c3 = 0;
			for(int j = i+1; j < info.size(); j++){
				if(info[j].type == "Iris-setosa")
					c1++;
				else if(info[j].type == "Iris-versicolor")
					c2++;
				else if(info[j].type == "Iris-virginica")
					c3++;
			}
			proC1 = ((float)c1 / (float)(info.size()-i-1));
			proC2 = ((float)c2 / (float)(info.size()-i-1));
			proC3 = ((float)c3 / (float)(info.size()-i-1));
			float P1 = (proC1 == 0) ? 0 : (proC1 * log(proC1) / log(2));
			float P2 = (proC2 == 0) ? 0 : (proC2 * log(proC2) / log(2));
			float P3 = (proC3 == 0) ? 0 : (proC3 * log(proC3) / log(2));
			float hi = (float)(info.size()-i-1)/(float)(info.size()) * (-(P1+P2+P3));
			tempRem = lw + hi;
			tempBound.rem = tempRem;
			// compute IG
			float ig = H - tempRem;
			tempBound.IG = ig;

			bound4.push_back(tempBound);
		}
	}
	if(bound4.size() != 0){
		sort(bound4.begin(), bound4.end(), compare5);
		Maxb[3] = bound4[0];
	}
	else{
		tempBound.rem = 0;
		tempBound.IG = H - tempBound.rem;
		tempBound.threshold = 0;
		bound4.push_back(tempBound);
		Maxb[3] = bound4[0];
	}
	//------------------------------

	float temp = 0.0;
	int MaxIndex;
	string anStr;
	float tempThre;
	for(int i = 0; i < 4; i++){
		if(Maxb[i].IG >= temp){
			temp = Maxb[i].IG;
			MaxIndex = i;
		}
	}

	if(MaxIndex == 0){
		anStr = "sepal_len";
		tempThre = Maxb[0].threshold;
	}
	else if(MaxIndex == 1){
		anStr = "sepal_wid";
		tempThre = Maxb[1].threshold;
	}
	else if(MaxIndex == 2){
		anStr = "petal_len";
		tempThre = Maxb[2].threshold;
	}
	else if(MaxIndex == 3){
		anStr = "petal_wid";
		tempThre = Maxb[3].threshold;
	}

	Classification c;
	c.feature = anStr;
	c.Threshold = tempThre;
	c.Ig = temp;
	if(H == c.Ig){
		c.same = true;
	}
	else
		c.same = false;
	return c;
}
//------------------------------------------------------------------


//----------------------BuildTree function--------------------------
void BuildTree(dtree root, vector<Flower> &f){
	dtree leftChild;
	dtree rightChild;
	
	if(root->ClassifyFeature.same){ // leave node
		root->left = NULL;
		root->right = NULL;
		root->leave = true;
		//cout << "leave" << endl;
		root->flowerType = f[0].type;
		return;
	}

	float thre = root->ClassifyFeature.Threshold;
	vector<Flower> f1,f2;

	if(root->ClassifyFeature.feature == "sepal_len"){
		for(int i = 0; i < f.size(); i++){
			if(f[i].sepal_len < thre){
				f1.push_back(f[i]);
			}
			else{
				f2.push_back(f[i]);
			}
		}
	}
	else if(root->ClassifyFeature.feature == "sepal_wid"){
		for(int i = 0; i < f.size(); i++){
			if(f[i].sepal_wid < thre){
				f1.push_back(f[i]);
			}
			else{
				f2.push_back(f[i]);
			}
		}
	}
	else if(root->ClassifyFeature.feature == "petal_len"){
		for(int i = 0; i < f.size(); i++){
			if(f[i].petal_len < thre){
				f1.push_back(f[i]);
			}
			else{
				f2.push_back(f[i]);
			}
		}
	}
	else if(root->ClassifyFeature.feature == "petal_wid"){
		for(int i = 0; i < f.size(); i++){
			if(f[i].petal_wid < thre){
				f1.push_back(f[i]);
			}
			else{
				f2.push_back(f[i]);
			}
		}
	}

	leftChild = new treenode; // build left subtree
	leftChild->ClassifyFeature = ComputeIG(f1);
	if(leftChild->ClassifyFeature.feature == "serious_error"){
		root->left = NULL;
		root->right = NULL;
		root->leave = true;
		root->flowerType = f[0].type;
		//cout << "sleave" << endl;
		return;
	}
	leftChild->leave = false;
	root->left = leftChild;
	BuildTree(root->left, f1);

	rightChild = new treenode; // build right subtree
	rightChild->ClassifyFeature = ComputeIG(f2);
	if(rightChild->ClassifyFeature.feature == "serious_error"){
		root->left = NULL;
		root->right = NULL;
		root->leave = true;
		root->flowerType = f[0].type;
		//cout << "sleave" << endl;
		return;
	}
	rightChild->leave = false;
	root->right = rightChild;
	BuildTree(root->right, f2);

}
//------------------------------------------------------------------

//---------------------TestData function----------------------------
Answer TestData(dtree root, vector<Flower> &f){
	Answer ans;
	string type;
	int rightNum = 0;
	int seNum = 0;
	int realSeNum = 0;
	int colorNum = 0;
	int realColorNum = 0;
	int caNum = 0;
	int realCaNum = 0;
	int r_seNum = 0;
	int r_realSeNum = 0;
	int r_colorNum = 0;
	int r_realColorNum = 0;
	int r_caNum = 0;
	int r_realCaNum = 0;
	float acc;
	float pre_se;
	float pre_col;
	float pre_ca;
	float rec_se;
	float rec_col;
	float rec_ca;

	for(int i = 0; i < f.size(); i++){
		dtree cur = root;

		while(!cur->leave){
			float thre = cur->ClassifyFeature.Threshold;

			if(cur->ClassifyFeature.feature == "sepal_len"){
				if(f[i].sepal_len < thre){
					cur = cur->left;
				}
				else{
					cur = cur->right;
				}
			}
			else if(cur->ClassifyFeature.feature == "sepal_wid"){
				if(f[i].sepal_wid < thre){
					cur = cur->left;
				}
				else{
					cur = cur->right;
				}
			}
			else if(cur->ClassifyFeature.feature == "petal_len"){
				if(f[i].petal_len < thre){
					cur = cur->left;
				}
				else{
					cur = cur->right;
				}
			}
			else if(cur->ClassifyFeature.feature == "petal_wid"){
				if(f[i].petal_wid < thre){
					cur = cur->left;
				}
				else{
					cur = cur->right;
				}
			}
		}

		if(f[i].type == cur->flowerType){
			rightNum++;
		}

		if(cur->flowerType == "Iris-setosa"){
			seNum++;
			if(f[i].type == "Iris-setosa")
				realSeNum++;
		}
		else if(cur->flowerType == "Iris-versicolor"){
			colorNum++;
			if(f[i].type == "Iris-versicolor")
				realColorNum++;
		}
		else if(cur->flowerType == "Iris-virginica"){
			caNum++;
			if(f[i].type == "Iris-virginica")
				realCaNum++;
		}


		if(f[i].type == "Iris-setosa"){
			r_seNum++;
			if(cur->flowerType == "Iris-setosa")
				r_realSeNum++;
		}
		else if(f[i].type == "Iris-versicolor"){
			r_colorNum++;
			if(cur->flowerType == "Iris-versicolor")
				r_realColorNum++;
		}
		else if(f[i].type == "Iris-virginica"){
			r_caNum++;
			if(cur->flowerType == "Iris-virginica")
				r_realCaNum++;
		}

	}
	// accuracy
	acc = (float)rightNum / (float)f.size();
	// precision
	pre_se = (seNum == 0) ? 0 : (float)realSeNum / (float)seNum;
	pre_col = (colorNum == 0) ? 0 : (float)realColorNum / (float)colorNum;
	pre_ca = (caNum == 0) ? 0 : (float)realCaNum / (float)caNum;
	// recall
	rec_se = (r_seNum == 0) ? 0 : (float)r_realSeNum / (float)r_seNum;
	rec_col = (r_colorNum == 0) ? 0 : (float)r_realColorNum / (float)r_colorNum;
	rec_ca = (r_caNum == 0) ? 0 : (float)r_realCaNum / (float)r_caNum;

	ans.Accuracy = acc;
	ans.Se_precision = pre_se;
	ans.Color_precision = pre_col;
	ans.Ca_precision = pre_ca;
	ans.Se_recall = rec_se;
	ans.Color_recall = rec_col;
	ans.Ca_recall = rec_ca;

	return ans;
}
//------------------------------------------------------------------
