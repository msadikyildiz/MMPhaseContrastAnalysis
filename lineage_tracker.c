#include <stdio.h>
#include <stdlib.h> 
#include <math.h>

#define max(a,b) ((a>b)?(a):(b))
#define min(a,b) ((a<b)?(a):(b))
#define INT_MAX 2e9

// roughly growth rate per pixel per frame
#define GROWTH_PER_FRAME 1
// Parameters of the DP Tracking Algorithm
// W_L: penalty for length difference
// W_O: score for overlap
// W_NA: penalty for skipping a cell
// W_MNA: penalty for skipping a cell via merging cells 
#define W_L		-10
#define W_O		1
#define W_NA	-50
#define W_MNA	-1e6

int n_frame;
int* n_vectors;
int* parents;
int** errors;
char** tracking;
int** posY0;
int** posY1;
int** vectors;
int** cell_ids;
int nextCellID=1;


int readVectors(char file[])
{
	int i,j,n_vector;
	FILE *f=fopen(file,"r");

	if(!f)
		return 1;

	char *buffer;
	size_t bufsize=100;
	buffer=(char *)malloc(bufsize * sizeof(char));

	fscanf(f,"%d,", &n_frame);
	getline(&buffer,&bufsize,f);\

	vectors=(int**)malloc(sizeof(int*)*n_frame);
	cell_ids=(int**)malloc(sizeof(int*)*n_frame);
	parents=(int*)calloc(sizeof(int),n_frame*10);
	errors=(int**)malloc(sizeof(int)*n_frame*10);
	tracking=(char**)malloc(sizeof(char*)*n_frame*10);
	posY0=(int**)malloc(sizeof(int*)*n_frame*10);
	posY1=(int**)malloc(sizeof(int*)*n_frame*10);
	n_vectors=(int*)malloc(sizeof(int)*n_frame);

	for(i=0;i<n_frame;i++)
	{
		fscanf(f,"%d,", &n_vectors[i]);
		vectors[i]=(int*)malloc(sizeof(int)*(n_vectors[i]<<1));
		cell_ids[i]=(int*)calloc(sizeof(int),(n_vectors[i]+1));
		for(j=0;j<(n_vectors[i]<<1);j++)
			fscanf(f,"%d,",&(vectors[i][j]));
		getline(&buffer,&bufsize,f);
	}

	for(i=0;i<n_frame*10;i++)
	{
		tracking[i]=(char*)calloc(sizeof(char),n_frame);
		posY0[i]=(int*)calloc(sizeof(int),n_frame);
		posY1[i]=(int*)calloc(sizeof(int),n_frame);
		errors[i]=(int*)calloc(sizeof(int),n_frame);
	}

	return 0;
}

void writeCellIDs(char file[])
{
	int i,j;
	FILE *f=fopen(file,"w");

	fprintf(f,"%d",n_frame);
	for(j=0;j<99;j++)
		fprintf(f,",");
	fprintf(f,"\n");

	for(i=0;i<n_frame;i++)
	{
		fprintf(f,"%d,",n_vectors[i]);
		for(j=1;j<=(n_vectors[i]);j++)
			fprintf(f,"%d,", cell_ids[i][j]);

		for(j=0;j<(98-n_vectors[i]);j++)
			fprintf(f,",");
		fprintf(f,"\n");
	}
	fclose(f);
}

void writeTrackingAnnot2(char file[])
{
	int i,j,k,l,daughters[2];
	FILE *f=fopen(file,"w");
	fprintf(f,"cellID,frameID,posY0,posY1,tracking,parentID,daughter1ID,daughter2ID,extraID\n");

	for(i=1;i<nextCellID;i++)
	{
		daughters[0]=0;
		daughters[1]=0;
		for(k=i+1,l=0;k<nextCellID;k++)
			if(parents[k]==i)
				daughters[l++]=k;

		for(j=0;j<n_frame-1;j++)
			if(tracking[i][j])
				fprintf(f,"%d,%d,%d,%d,%d,%d,%d,%d,%d\n",i,j+1,posY0[i][j],posY1[i][j],
						tracking[i][j],parents[i],daughters[1],daughters[0],errors[i][j]);

	}
	fclose(f);
}


// evaluates the length difference penalty score of two-vectors one-to-one
int L(int v1i, int i, int v2i, int j)
{
	int *v1=vectors[v1i], *v2=vectors[v2i];
	return W_L * abs( GROWTH_PER_FRAME+(v1[(i<<1)+1]-v1[i<<1]) - (v2[(j<<1)+1]-v2[j<<1]) );
}

// evaluates the length difference penalty score of three-vectors one-to-two (split event)
int L2(int v1i, int i, int v2i, int j, int k)
{
	int *v1=vectors[v1i], *v2=vectors[v2i];
	return W_L * abs( GROWTH_PER_FRAME*2+(v1[(i<<1)+1]-v1[i<<1]) - (v2[(j<<1)+1]-v2[j<<1])
											  - (v2[(k<<1)+1]-v2[k<<1]) );
}

// evaluates the length difference penalty score of four-vectors one-to-three (3split event)
int L3(int v1i, int i, int v2i, int j, int k, int l)
{
	int *v1=vectors[v1i], *v2=vectors[v2i];
	return W_L * abs( GROWTH_PER_FRAME*3+(v1[(i<<1)+1]-v1[i<<1]) - (v2[(j<<1)+1]-v2[j<<1])
											  - (v2[(k<<1)+1]-v2[k<<1])
											  - (v2[(l<<1)+1]-v2[l<<1]));
}

// evaluates the y-position overlapping scores of two vectors one-to-one
int O(int v1i, int i, int v2i, int j)
{
	int v1_st=vectors[v1i][i<<1], v1_en=vectors[v1i][(i<<1)+1];
	int v2_st=vectors[v2i][j<<1], v2_en=vectors[v2i][(j<<1)+1];
	return W_O * (min(v2_en,v1_en)-max(v2_st,v1_st));
}

void trackVectors()
{
	int i,j,match,split,split3,merge2,merge3,na,na_entry;
	int **H;
	int **D;
	int *v1; int v1_n;
	int *v2; int v2_n;
	int v1i=5, v2i=6;

	// initialize cell IDs of the first frame
	for(i=1;i<=n_vectors[0];i++)
		cell_ids[0][i]=nextCellID++;


	for(v1i=0,v2i=1; v2i<n_frame-1; v1i++,v2i++)
	{

		// initialize iterators
		v1=vectors[v1i];
		v2=vectors[v2i];
		v1_n=n_vectors[v1i];
		v2_n=n_vectors[v2i];

		// create DP-score matrix
		H=(int**)malloc(sizeof(int*)*(v1_n+1));
		for(i=0;i<v1_n+1;i++) H[i]=(int*)calloc(sizeof(int),(v2_n+1));

		// create DP-direction matrix
		// 1 => match 2 => split 3=> na
		D=(int**)malloc(sizeof(int*)*(v1_n+1));
		for(i=0;i<v1_n+1;i++) D[i]=(int*)calloc(sizeof(int),(v2_n+1));

		// initiate first column of the matrix
		for(i=0;i<=v1_n;i++)
			H[i][0]=i*W_NA;
		for(i=0;i<=v2_n;i++)
			H[0][i]=i*W_NA;

		// fill matrix
		for(i=1;i<=v1_n;i++)
		{
			for(j=1;j<=v2_n;j++)
			{
				// match
				match=H[i-1][j-1]+L(v1i,i-1,v2i,j-1)+O(v1i,i-1,v2i,j-1);

				// split
				if(j>1)
					split=H[i-1][j-2]+L2(v1i,i-1,v2i,j-1,j-2);
									 //+O(v1i,i-1,v2i,j-1)+O(v1i,i-1,v2i,j-2);
				else
					split=-INT_MAX;

				// split3
				if(j>2)
					split3=H[i-1][j-3]+L3(v1i,i-1,v2i,j-1,j-2,j-3)
									 +O(v1i,i-1,v2i,j-1)+O(v1i,i-1,v2i,j-2)
									 +O(v1i,i-1,v2i,j-3);
				else
					split3=-INT_MAX;

				// merge2
				if(i>1)
					merge2=H[i-2][j-1]+2*L2(v2i,j-1,v1i,i-1,i-2);
									 //+O(v1i,i-1,v2i,j-1)+O(v1i,i-1,v2i,j-2);
				else
					merge2=-INT_MAX;

				// evalute the score of one cell going AWOL
				merge3=-INT_MAX;
				if(j==1 || j==v2_n || D[i-1][j]==5)
					na=H[i-1][j]+W_NA;
				else
					na=H[i-1][j]+W_MNA;

				// evalute the score of one cell appearing out of blue
				if(i==1 || i==v1_n || D[i][j-1]==6)
					na_entry=H[i][j-1]+W_NA;
				else
					na_entry=H[i][j-1]+W_MNA;

				// make the move with highest score
				if(match >= split && match >= na && match >=na_entry && match >= merge2
					&& match >= split3)
					H[i][j]=match, D[i][j]=1;
				else if(split >= match && split >= na && split >=na_entry && split >= merge2
						&& split >= split3)
					H[i][j]=split, D[i][j]=2;
				else if(split3 >= match && split3 >= na && split3 >=na_entry && split3 >= merge2
						&& split3 >= split)
					H[i][j]=split3, D[i][j]=3;
				else if(merge2 >= match && merge2 >= na && merge2 >=na_entry && merge2 >= split
						&& merge2 >= split3)
					H[i][j]=merge2, D[i][j]=4;
				else if(na >= match && na >= split && na >=na_entry && na >= merge2
						&& na >= split3)
					H[i][j]=na, D[i][j]=5;
				else if(na_entry >= match && na_entry >= split && na_entry >=na && na_entry >= merge2
						&& na_entry >= split3)
					H[i][j]=na_entry, D[i][j]=6;
			}
		}

		// assign unique cell IDs
		i=v1_n, j=v2_n;
		while(i>=1 && j>=1)
		{
			switch(D[i][j])
			{
				case 1: // match
					cell_ids[v2i][j]=cell_ids[v1i][i];
					tracking[cell_ids[v2i][j]][v2i]=1;
					posY0[cell_ids[v2i][j]][v2i]=vectors[v2i][(j-1)<<1];
					posY1[cell_ids[v2i][j]][v2i]=vectors[v2i][((j-1)<<1)+1];
					i--,j--;
					break;
				case 2: // split
					parents[nextCellID]=cell_ids[v1i][i];
					cell_ids[v2i][j]=nextCellID++;
					parents[nextCellID]=cell_ids[v1i][i];
					cell_ids[v2i][j-1]=nextCellID++;
					tracking[cell_ids[v2i][j]][v2i]=2;
					tracking[cell_ids[v2i][j-1]][v2i]=2;

					posY0[cell_ids[v2i][j]][v2i]=vectors[v2i][(j-1)<<1];
					posY1[cell_ids[v2i][j]][v2i]=vectors[v2i][((j-1)<<1)+1];

					posY0[cell_ids[v2i][j-1]][v2i]=vectors[v2i][(j-2)<<1];
					posY1[cell_ids[v2i][j-1]][v2i]=vectors[v2i][((j-2)<<1)+1];

					j-=2; i-=1;
					break;
				case 3: // split3
					cell_ids[v2i][j]=cell_ids[v1i][i];
					cell_ids[v2i][j-1]=nextCellID++;
					cell_ids[v2i][j-2]=nextCellID++;
					tracking[cell_ids[v2i][j]][v2i]=3;
					tracking[cell_ids[v2i][j-1]][v2i]=3;
					tracking[cell_ids[v2i][j-2]][v2i]=3;
					posY0[cell_ids[v2i][j]][v2i]=vectors[v2i][(j-1)<<1];
					posY1[cell_ids[v2i][j]][v2i]=vectors[v2i][((j-1)<<1)+1];
					posY0[cell_ids[v2i][j-1]][v2i]=vectors[v2i][(j-2)<<1];
					posY1[cell_ids[v2i][j-1]][v2i]=vectors[v2i][((j-2)<<1)+1];
					posY0[cell_ids[v2i][j-2]][v2i]=vectors[v2i][(j-3)<<1];
					posY1[cell_ids[v2i][j-2]][v2i]=vectors[v2i][((j-3)<<1)+1];
					j-=3; i-=1;
					break;
				case 4: // merge2
					cell_ids[v2i][j]=cell_ids[v1i][i-1];
					tracking[cell_ids[v2i][j]][v2i]=4;
					errors[cell_ids[v2i][j]][v2i]=cell_ids[v1i][i];
					posY0[cell_ids[v2i][j]][v2i]=vectors[v2i][(j-1)<<1];
					posY1[cell_ids[v2i][j]][v2i]=vectors[v2i][((j-1)<<1)+1];
					j-=1; i-=2;
					break;
				case 5: // NA
					tracking[cell_ids[v1i][i]][v2i]=5;
					i--;
					break;
				case 6: // NA-entry
					cell_ids[v2i][j]=nextCellID++;
					tracking[cell_ids[v2i][j]][v2i]=6;
					j--;
					break;
			}
		}

		// if(v1i==803)
		// {
		// printf("%d %d\n",v1_n,v2_n);
		// for(i=0;i<=v1_n;i++)
		// {
		// 	for(j=0;j<=v2_n;j++)
		// 		printf("%d\t",H[i][j]);
		// 	printf("\n");
		// }

		// printf("\n");

		// for(i=0;i<=v1_n;i++)
		// {
		// 	for(j=0;j<=v2_n;j++)
		// 		printf("%d\t",D[i][j]);
		// 	printf("\n");
		// }
		// }

		// free memory
		// for(i=0;i<v1_n+1;i++)
		// 	free(H[i]);
		// free(H);
		
		// for(i=0;i<v1_n+1;i++)
		// 	free(D[i]);
		// free(D);
	}

}

int main()
{
	int pos=5;
	int i,j;
	char file[100];
	char cellIDFile[100];
	char trackingFile[100];
	
	for(i=1;i<=15;i++)
	{
		sprintf(file,"/Users/sadik/Desktop/pombe2/Annotated_Kymographs/XY%02d/kymo%02d_vectors.csv",pos,i);
		sprintf(cellIDFile,"/Users/sadik/Desktop/pombe2/Annotated_Kymographs/XY%02d/kymo%02d_cellIDs.csv",pos,i);
		sprintf(trackingFile,"/Users/sadik/Desktop/pombe2/Annotated_Kymographs/XY%02d/kymo%02d_tracking.csv",pos,i);
		nextCellID=1;
		
		if (readVectors(file))
		{
			printf("Channel%02d is not found.\n",i);
			continue;
		}
		// printf("read ok.\n");
		trackVectors();
		// printf("tracking ok.\n");
		writeCellIDs(cellIDFile);
		writeTrackingAnnot2(trackingFile);

		// free memory for next lane
		for(j=0;j<n_frame;j++)
			free(vectors[j]),free(cell_ids[j]);
		free(vectors);
		free(cell_ids);
		free(parents);
		free(errors);
		free(n_vectors);

		printf("Channel%02d: Lineage tracking successful.\n",i);
	}

	return 0;	
}