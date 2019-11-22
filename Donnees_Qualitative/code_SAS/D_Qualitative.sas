PROC IMPORT OUT= WORK.Primary_tumor_data 
            DATAFILE= "C:\Users\FOUTSE\Desktop\D_Qualitatives\Primary_tumor_data.txt" 
            DBMS=TAB REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;

proc contents data =Primary_tumor_data; run;
* 2)pour les variables quantitatives : moyenne variance, covariance, corrélation,  faire un histogramme, box plot...;

proc means data =Primary_tumor_data mean;OUTPUT OUT =MOY; run; proc print; run;

data moy_seule; set moy; if _stat_ = 'MEAN'; drop _type_  _freq_ ; run; proc print; run;

proc univariate data =Primary_tumor_data; histogram; run;

proc corr data =Primary_tumor_data cov; run;

*représentation graphique des var qualitatives;

PROC GCHART DATA=Primary_tumor_data;
VBAR age ;
VBAR sex;
VBAR bone;
VBAR Bone_marrow;
VBAR lung;
VBAR pleura;
VBAR peritoneum;
VBAR liver;
VBAR brain;
VBAR skin;
VBAR neck;
VBAR supraclavicular;
VBAR axillar;
VBAR mediastinum;
VBAR abdominal
class;
RUN;

*ACM sur les variables qualitatives on met species en supplémentaire  il faut utiliser option binary pour avoir les coordonnées des individus;

PROC CORRESP DATA =Primary_tumor_data binary  n=15 outc= sortie; 
TABLES class age sex bone Bone_marrow lung pleura peritoneum liver brain skin neck supraclavicular  axillar mediastinum abdominal;
supplementary class;
RUN;

PROC CORRESP DATA =Primary_tumor_data binary  n=5 outc= sortie; 
TABLES class age sex bone Bone_marrow;
supplementary class;
RUN;
