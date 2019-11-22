PROC IMPORT OUT= WORK.GMODataAcmred1
            DATAFILE= "C:\Users\FOUTSE\Desktop\D_Qualitatives\GMODataAcmred1.txt" 
            DBMS=TAB REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;


proc contents data =GMODataAcmred1; run;

proc print data = GMODataAcmred1; run;

*Analyse uni dimensionelle*;
PROC GCHART DATA=GMODataAcmred1;
*PIE Implicated Position_Culture Position_Al_H Position_Al_A Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Grandparents Sex Age Profession Relation Political_Party / slice=outside percent=outside;
vbar Implicated Position_Culture Position_Al_H Position_Al_A Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Grandparents Sex Age Profession Relation Political_Party /  raxis=axis1 gaxis=axis2; 
run;

*Analyse bi-dimensionelle;
proc freq data = GMODataAcmred1;
table Implicated*(Position_Culture Position_Al_H Position_Al_A Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Grandparents Sex Age Profession Relation Political_Party) /expected cellchi2 ; *CHISQ FAIT LES TESTS;

run;

proc corresp data = GMODataAcmred1 mca dim=2 ;
tables Position_Culture Position_Al_H Position_Al_A Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Grandparents Sex Age Profession Relation Political_Party Implicated;
supplementary Implicated;
run;

* 53 modalitee de variable explicative - 18 variable explicatives******************************;

proc corresp data = GMODataAcmred1 binary dim=35 outc=sortie noprint ;
tables Implicated Position_Culture Position_Al_H Position_Al_A Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Grandparents Sex Age Profession Relation Political_Party ;
supplementary Implicated;
run;






*SORTIE CONTIENT  DES INFORMATIONS AUTRES QUE SUR LES INDIVIDUS ON LES SUPPRIME ET ON GARDE LES COORDONNEES DE 1 à NOMBRE_TOT_DE_DIM  (ICI=24);
data resultat_acm; 
set sortie;
keep DIM1-DIM35;
if _TYPE_ ='OBS';
run;

*********************************************CREATION DE NOUVEAU FICHIER*************************************
*************************************************************************************************************;

data sortiesup;
set GMODataAcmred1;
keep Implicated;
run;

*On merge la variable a expliquer avec les composantes et on le sauvegarde dans sortieACM;

data sortieACM;
merge sortiesup resultat_acm;
run;

*****************************************************************************************************************************************************************************
********************************************************************************ANALYSE DISCRIMINANTE FACTORIELLES***********************************************************
*****************************************************************************************************************************************************************************;

*Analyse factoriellle discriminante sur variable qualitative AFD DISQUAL;
proc candisc data =sortieACM;
class Implicated;
var dim1-dim35;
run;

* DISCRIMINATION BAYESIENNE ON CALCULE LE % DE MAL CLASSES PAR VALIDATION CROIS2E;
PROC DISCRIM DATA= sortieACM  all crossvalidate canonical;
class Implicated;
var dim1-dim35;
run;

proc discrim data=sortieACM method = npar k=5 crossvalidate canonical ;
class Implicated;
var DIM1-DIM35 ;
run;

*Cette fonction nous montre les dimentions les plus importante et on les utilisera pour pouvoir faire l'analyse discriminante*
******************************************************************************************************************************;
PROC STEPDISC DATA=sortieACM sw SHORT;
class Implicated; 
var dim1-dim35;
run;




*****************************************************************************************************************************************************************************
********************************************************************************CLASSIFICATION*******************************************************************************
*****************************************************************************************************************************************************************************;

proc cluster data=sortieACM method=ward out=ARBRE;
var DIM1-DIM35;
run;
proc tree data=ARBRE out=PLAN nclusters=2 ;
COPY DIM1 DIM2;
run;

proc sgplot data=PLAN;
   scatter y=dim2 x=dim1 / group=cluster;
run;
