#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <algorithm>
#include <math.h>
#define INSTANCESIZE 300
#define TRAINSIZE 264
#define TESTSIZE 36

using namespace std;

int sortIndex; // for compare() to identify the index of attribute

class Ecoli{
public:
	int index;
	char name[20];
	float attribute[9];
	char classes[5];
	float recordDistance;
};

struct KNode{
	Ecoli nodeThreshold;
	int depth;

	struct KNode *parent;
	struct KNode *left;
	struct KNode *right;
};
typedef struct KNode Knode;
typedef Knode *knode;

Ecoli Compare_biggest_variance(vector<Ecoli> &e);
void Build_KDtree(knode root, vector<Ecoli> &e, int Depth);
knode DescendToLeaf(knode cur, Ecoli e);
float Compute_distance(Ecoli e1, Ecoli e2);
Ecoli KNN_1(knode tree, Ecoli test);
vector<Ecoli> KNN_n(knode tree, Ecoli test, int k);
knode Back_to_parent_and_prune(knode cur);
float Compute_Boundary_distance(knode cur, Ecoli test);
bool Voting_and_compareTest(vector<Ecoli> &e, Ecoli test);

bool compare(const Ecoli &e1, const Ecoli &e2){
	return (e1.attribute[sortIndex] < e2.attribute[sortIndex]);
}

bool compare_dis(const Ecoli &e1, const Ecoli &e2){
	return (e1.recordDistance < e2.recordDistance);
}

int main(){
	FILE *fp;

	fp = fopen("train.csv", "r");
	if(!fp){
		printf("Can't open the file!\n");
		return 0;
	}

	char Ignore[100];
	fscanf(fp, "%s", Ignore);

	int index;
	Ecoli eco[INSTANCESIZE];

	for(int i = 0; i < INSTANCESIZE; i++){
		fscanf(fp, "%d,%[^','],%f,%f,%f,%f,%f,%f,%f,%f,%f,%s", &eco[i].index, eco[i].name, &eco[i].attribute[0], &eco[i].attribute[1], &eco[i].attribute[2], &eco[i].attribute[3], &eco[i].attribute[4], &eco[i].attribute[5], &eco[i].attribute[6], &eco[i].attribute[7], &eco[i].attribute[8], eco[i].classes);
	}
	vector<Ecoli> Eco, testEco;
	Eco.resize(TRAINSIZE);
	testEco.resize(TESTSIZE);
	for(int i = 0; i < TRAINSIZE; i++){
		Eco[i] = eco[i];
	}
	for(int i = TRAINSIZE; i < INSTANCESIZE; i++){
		testEco[i-TRAINSIZE] = eco[i];
	}
	//----------------------Build KD-tree----------------------
	knode Root;
	Root = new Knode;
	Root->parent = NULL;
	Build_KDtree(Root, Eco, 0);
	//---------------------------------------------------------
	FILE *fpw;

	fpw = fopen("output.txt", "w");
	//-----------------------KNN : K = 1-----------------------
	Ecoli k1[TESTSIZE];
	for(int i = 0; i < TESTSIZE; i++){
		k1[i] = KNN_1(Root, testEco[i]);
		Build_KDtree(Root, Eco, 0);
	}
	int success = 0;
	for(int i = 0; i < TESTSIZE; i++){
		if(!strcmp(k1[i].classes, testEco[i].classes)) success++;
	}
	float accuracy = ((float)success / TESTSIZE);
	// print to screen
	cout << "KNN accuracy: " << accuracy << endl;
	for(int i = 0; i < 3; i++) cout << k1[i].index << endl;
	// print to output.txt
	fprintf(fpw, "KNN accuracy: %f\n", accuracy);
	for(int i = 0; i < 3; i++) fprintf(fpw, "%d\n", k1[i].index);
	//---------------------------------------------------------
	cout << endl;
	fprintf(fpw, "\n");
	//-----------------------KNN : K = 5-----------------------
	vector<vector<Ecoli> > k5;
	k5.resize(TESTSIZE);
	for(int i = 0; i < TESTSIZE; i++){
		k5[i] = KNN_n(Root, testEco[i], 5);
		Build_KDtree(Root, Eco, 0);
	}
	success = 0;
	for(int i = 0; i < TESTSIZE; i++){
		if(Voting_and_compareTest(k5[i], testEco[i])) success++;
	}
	accuracy = ((float)success / TESTSIZE);
	cout << "KNN accuracy: " << accuracy << endl;
	fprintf(fpw, "KNN accuracy: %f\n", accuracy);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < k5[i].size(); j++){
			cout << k5[i][j].index << " ";
			fprintf(fpw, "%d ", k5[i][j].index);
		}
		cout << endl;
		fprintf(fpw, "\n");
	}
	//---------------------------------------------------------
	cout << endl;	
	fprintf(fpw, "\n");	
	//-----------------------KNN : K = 10----------------------
	vector<vector<Ecoli> > k10;
	k10.resize(TESTSIZE);
	for(int i = 0; i < TESTSIZE; i++){
		k10[i] = KNN_n(Root, testEco[i], 10);
		Build_KDtree(Root, Eco, 0);
	}
	success = 0;
	for(int i = 0; i < TESTSIZE; i++){
		if(Voting_and_compareTest(k10[i], testEco[i])) success++;
	}
	accuracy = ((float)success / TESTSIZE);
	cout << "KNN accuracy: " << accuracy << endl;
	fprintf(fpw, "KNN accuracy: %f\n", accuracy);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < k10[i].size(); j++){
			cout << k10[i][j].index << " ";
			fprintf(fpw, "%d ", k10[i][j].index);
		}
		cout << endl;
		fprintf(fpw, "\n");
	}
	//---------------------------------------------------------
	cout << endl;
	fprintf(fpw, "\n");
	//-----------------------KNN : K = 100---------------------
	vector<vector<Ecoli> > k100;
	k100.resize(TESTSIZE);
	for(int i = 0; i < TESTSIZE; i++){
		k100[i] = KNN_n(Root, testEco[i], 100);
		Build_KDtree(Root, Eco, 0);
	}
	success = 0;
	for(int i = 0; i < TESTSIZE; i++){
		if(Voting_and_compareTest(k100[i], testEco[i])) success++;
	}
	accuracy = ((float)success / TESTSIZE);
	cout << "KNN accuracy: " << accuracy << endl;
	fprintf(fpw, "KNN accuracy: %f\n", accuracy);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < k100[i].size(); j++){
			cout << k100[i][j].index << " ";
			fprintf(fpw, "%d ", k100[i][j].index);
		}
		cout << endl;
		fprintf(fpw, "\n");
	}
	//----------------------------------------------------------

	return 0;
}

Ecoli Compare_biggest_variance(vector<Ecoli> &e){ // sort the data(e) by the biggest variance attribute 
	Ecoli returnValue;							  // and return median ecoli
	int attributeIndex;
	float tempVar = 0;
	for(int i = 0; i < 9; i++){
		float average;
		float sum = 0.0;
		for(int j = 0; j < e.size(); j++){
			sum += e[j].attribute[i];
		}
		average = (sum/(float)e.size());
		float variance;
		float sum2 = 0.0;
		for(int j = 0; j < e.size(); j++){
			sum2 += ((e[j].attribute[i] - average) * (e[j].attribute[i] - average));
		}
		variance = (sum2/(float)e.size());
		if(variance > tempVar){
			attributeIndex = i;
			tempVar = variance;
		}
	}
	sortIndex = attributeIndex;
	sort(e.begin(), e.end(), compare);
	returnValue = e[(int)(e.size() / 2)];
	return returnValue;
}

void Build_KDtree(knode root, vector<Ecoli> &e, int Depth){
	int median = (e.size() / 2);

	root->nodeThreshold = Compare_biggest_variance(e); // pivot node store in current node
	root->depth = Depth;
	if(!median) return; // only one ecoli , this node is leaf node
	vector<Ecoli> e1, e2;
	e1.clear();
	e2.clear();
	for(int i = 0; i < median; i++){ // set less than median by comparing attribute
		e1.push_back(e[i]);
	}
	for(int i = median + 1; i < e.size(); i++){ // set more than median by comparing attribute
		e2.push_back(e[i]);
	}

	// leftchild
	knode leftchild = new Knode;
	root->left = leftchild; // connect child to parent
	leftchild->parent = root;
	leftchild->left = NULL; // we have to let leaf node's leftChild and rightChild be NULL , or there will be core dump
	leftchild->right = NULL;
	Build_KDtree(leftchild, e1, Depth + 1);

	// rightchild
	if(!e2.size()) return;
	knode rightchild = new Knode;
	root->right = rightchild;
	rightchild->parent = root;
	rightchild->left = NULL;
	rightchild->right = NULL;
	Build_KDtree(rightchild, e2, Depth + 1);
}

knode DescendToLeaf(knode cur, Ecoli e){ // traverse to leaf node (must arrive leaf node)
	while(cur->right || cur->left){
		int attIndex;
		attIndex = (cur->depth % 9);

		if(e.attribute[attIndex] < cur->nodeThreshold.attribute[attIndex]){
			if(!cur->left) cur = cur->right; // if there no leftChild existing , then go to rightChild
			else cur = cur->left;
		}
		else{
			if(!cur->right) cur = cur->left; // if there no rightChild existing , then go to leftChild
			else cur = cur->right;
		}
	}
	return cur;
}

float Compute_distance(Ecoli e1, Ecoli e2){ // compute two Ecoli's distance
	float dis;
	float sum = 0.0;
	for(int i = 0; i < 9; i++){
		sum += ((e1.attribute[i] - e2.attribute[i])*(e1.attribute[i] - e2.attribute[i]));
	}
	dis = sqrt(sum);
	return dis;
}

Ecoli KNN_1(knode tree, Ecoli test){ // for K = 1
	knode cur;
	cur = tree;
	cur = DescendToLeaf(cur, test); // first , traverse to leaf node

	Ecoli neighbor; // nearest neighbor
	float minDist = 100.0;
	float dis;
	
	while(cur){
		if(!cur->left && !cur->right){ // if current node is leaf node
			dis = Compute_distance(cur->nodeThreshold, test); // compute distance between current node's Ecoli and testdata
			if(dis < minDist && dis){ // update minimum distance and nearest neighbor if find shorter distance (not equal to itself)
				minDist = dis;
				neighbor = cur->nodeThreshold;
			}
			cur = Back_to_parent_and_prune(cur); // after comparing , go back to parent and prune this path
		}
		else{ // if current node is non-leaf node
			float check = Compute_Boundary_distance(cur, test); // compute the distance between this node's attribute boundary and testdata first
			if(check < minDist){ // if check is shorter than minimum distance , then descend to leaf node again
				cur = DescendToLeaf(cur, test);
			}
			else{ // otherwise , go back to parent and prune this path
				cur = Back_to_parent_and_prune(cur);
			}
		}
	}
	return neighbor;
}

vector<Ecoli> KNN_n(knode tree, Ecoli test, int k){ // for K > 1
	knode cur;
	cur = tree;
	cur = DescendToLeaf(cur, test);

	vector<Ecoli> Knearest; // store K nearest node
	Knearest.clear();
	float dis;
	float boundDis; // the maximum distance in Knearest
	Ecoli forPush;

	while(cur){
		if(Knearest.size() < k){ // first we traverse k node(from leaf node) and push them in Knearest
			if(!cur->left && !cur->right){ // leaf node
				dis = Compute_distance(cur->nodeThreshold, test);
				if(dis){ // remove node itself
					forPush = cur->nodeThreshold;
					forPush.recordDistance = dis;
					Knearest.push_back(forPush);
					cur = Back_to_parent_and_prune(cur); // after push_back , go to parent and keep traverse
					sort(Knearest.begin(), Knearest.end(), compare_dis); // sort the vector in ascending
					boundDis = Knearest[Knearest.size() - 1].recordDistance;
				}
			}
			else{ // non-leaf node
				cur = DescendToLeaf(cur, test);
			}
		}
		else{ // after collect k node , then keep traversing and find shorter distance
			if(!cur->left && !cur->right){
				dis = Compute_distance(cur->nodeThreshold, test);
				if(dis < boundDis && dis){ // if find shorter distance
					forPush = cur->nodeThreshold;
					forPush.recordDistance = dis;
					Knearest[Knearest.size() - 1] = forPush; // update Knearest
					sort(Knearest.begin(), Knearest.end(), compare_dis);
					boundDis = Knearest[Knearest.size() - 1].recordDistance;
				}
				cur = Back_to_parent_and_prune(cur);
			}
			else{
				float check = Compute_Boundary_distance(cur, test);
				if(check < boundDis){
					cur = DescendToLeaf(cur, test);
				}
				else{
					cur = Back_to_parent_and_prune(cur);
				}
			}
		}
	}
	return Knearest;
}

knode Back_to_parent_and_prune(knode cur){
	if(!cur->parent){ // current node is Root node
		cur = cur->parent;
		return cur;
	}

	if(cur->parent->left == cur){ // current node is parent's leftChild
		cur = cur->parent;		  // go back to parent
		cur->left = NULL;		  // prune this path out
	}
	else{ // current node is parent's rightChild
		cur = cur->parent;
		cur->right = NULL;
	}
	return cur;
}

float Compute_Boundary_distance(knode cur, Ecoli test){ // if traversing to non-leaf node , compute the distance 
	int attributeIndex = cur->depth;					// between test and current node by current attribute dimension
	float distance;

	if(cur->nodeThreshold.attribute[attributeIndex] > test.attribute[attributeIndex])
		distance = cur->nodeThreshold.attribute[attributeIndex] - test.attribute[attributeIndex];
	else
		distance = test.attribute[attributeIndex] - cur->nodeThreshold.attribute[attributeIndex];

	return distance;
}

bool Voting_and_compareTest(vector<Ecoli> &e, Ecoli test){
	char classes[8][5];
	strcpy(classes[0], "cp");
	strcpy(classes[1], "im");
	strcpy(classes[2], "pp");
	strcpy(classes[3], "imU");
	strcpy(classes[4], "om");
	strcpy(classes[5], "omL");
	strcpy(classes[6], "inL");
	strcpy(classes[7], "imS");
	int c[8];
	for(int i = 0; i < 8; i++) c[i] = 0;

	for(int i = 0; i < e.size(); i++){
		if(!strcmp(e[i].classes, classes[0])) c[0]++;
		else if(!strcmp(e[i].classes, classes[1])) c[1]++;
		else if(!strcmp(e[i].classes, classes[2])) c[2]++;
		else if(!strcmp(e[i].classes, classes[3])) c[3]++;
		else if(!strcmp(e[i].classes, classes[4])) c[4]++;
		else if(!strcmp(e[i].classes, classes[5])) c[5]++;
		else if(!strcmp(e[i].classes, classes[6])) c[6]++;
		else if(!strcmp(e[i].classes, classes[7])) c[7]++;
	}
	int temp = 0; // select the maximum number of classes
	for(int i = 0; i < 8; i++){
		if(c[i] > temp) temp = c[i];
	}
	if(temp == c[0]){
		if(!strcmp(test.classes, "cp")) return true;
		else return false;
	}
	else if(temp == c[1]){
		if(!strcmp(test.classes, "im")) return true;
		else return false;
	}
	else if(temp == c[2]){
		if(!strcmp(test.classes, "pp")) return true;
		else return false;
	}
	else if(temp == c[3]){
		if(!strcmp(test.classes, "imU")) return true;
		else return false;
	}
	else if(temp == c[4]){
		if(!strcmp(test.classes, "om")) return true;
		else return false;
	}
	else if(temp == c[5]){
		if(!strcmp(test.classes, "omL")) return true;
		else return false;
	}
	else if(temp == c[6]){
		if(!strcmp(test.classes, "inL")) return true;
		else return false;
	}
	else if(temp == c[7]){
		if(!strcmp(test.classes, "imS")) return true;
		else return false;
	}
}
