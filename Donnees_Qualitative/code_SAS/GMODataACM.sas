
PROC IMPORT OUT= WORK.GMODataRed
            DATAFILE= "C:\Users\FOUTSE\Desktop\D_Qualitatives\GMODataRed.txt" 
            DBMS=TAB REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;

proc contents data =GMODataRed; run;

PROC GCHART DATA=GMODataRed;
PIE Implicated Position_Culture Position_Al_H Position_Al_A Protest Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Useless_practice Grandparents Sex Age Profession Relation Political_Party / slice=outside percent=outside;
*vbar class age sex bone Bone_marrow lung pleura peritoneum liver brain skin neck /  raxis=axis1 gaxis=axis2; 
run;

proc freq data = GMODataRed;
table Implicated*(Position_Culture Position_Al_H Position_Al_A Protest Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Useless_practice Grandparents Sex Age Profession Relation Political_Party) /expected cellchi2 ; *CHISQ FAIT LES TESTS;
 
run;

proc corresp data = GMODataRed mca dim=2 ;
tables Position_Culture Position_Al_H Position_Al_A Protest Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Useless_practice Grandparents Sex Age Profession Relation Political_Party Implicated;
supplementary Implicated;
run;

* 57 modalitee de variable explicative - 20 variable explicatives******************************;

proc corresp data = GMODataRed binary dim=37 outc=sortie  ;
tables Implicated Position_Culture Position_Al_H Position_Al_A Protest Media_Passive Info_Active Phytosanitary_products Hunger Animal_feed Future_Progress Danger Threat Finan_risk Useless_practice Grandparents Sex Age Profession Relation Political_Party ;
supplementary Implicated;
run;

*****************************************************************************************************************************************************************************
********************************************************************************CLASSIFICATION*******************************************************************************
*****************************************************************************************************************************************************************************;

proc cluster data=sortie method=ward out=ARBRE;
var DIM1-DIM37;
run;
proc tree data=ARBRE out=PLAN nclusters=2 ;
COPY DIM1 DIM2;
run;

proc sgplot data=PLAN;
   scatter y=dim2 x=dim1 / group=cluster;
run;
