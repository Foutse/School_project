*IMPORTATION DES DONNEES  apres datafile il faut indiquer le chemin où se trouve le fichier texte à importer;

PROC IMPORT OUT= WORK.fichier_donnees 
            DATAFILE= "C:\NIANG\tried\fichier_donnees.txt"  
            DBMS=TAB REPLACE;
     GETNAMES=YES;  * la premiere ligne contient le nom des variables;
     DATAROW=2; 
RUN;

* pour faire des représentations graphiques;

PROC GCHART DATA=fichier_donnees;
PIE LISTE DES VARIABLES / slice=outside percent=outside;
vbar LISTE DES VARIABLES  /  raxis=axis1 gaxis=axis2; 
run;

/*La proc greplay  permet d'afficher plusieurs graphs sur une même page  voir help si nécessaire/ 

/* proc freq  pour etude uni et bidimensionnelle de données qualitatives , ici on etudie la liaison entre la var entre la var_qual_a_expliquer et toutes les autre*/

proc freq data = fichier_donnees;
table var_qual_a_expliquer*_ALL_ /expected cellchi2  chisq; *CHISQ FAIT LES TESTS;
run; 

* COMPARER AVEC :;
proc freq data = fichier_donnees;
table var_qual_a_expliquer*_ALL_ /expected cellchi2 ; *CHISQ FAIT LES TESTS;
 
run; 

*ACM SUR LES VAR QUALI SANS var_qual_a_expliquer. Par défaut on a les deux premiers axes ie dim=2, 
avec option mca  vous aurez que les resultats (graphiques en particulier) des modalités des variables on fait ACM du tableau de BURT
avec binary au lieu de mca on fait ACM du tableau disjonctif, 
on a alors les résultats pour les variables et les individus, LES PREMIERS PLANS A INTERPRETER 
ensuite vous pouvez refaire l'acm avec dim=3 ou 4 si vous avez retenu plus que 2 axes
;

ods graphics on;

proc corresp data = fichier_donnees mca dim=2 outc=sortie ;
tables LISTE DE TOUTES VARIABLES Y COMPRIS  var_qual_a_expliquer;
supplementary var_qual_a_expliquer;
run;
proc corresp data = fichier_donnees binary dim=2 ;
tables LISTE DE TOUTES VARIABLES Y COMPRIS  var_qual_a_expliquer;
supplementary var_qual_a_expliquer;
run;

 *SI VOUS SOUHAITER RECUPERER TOUTES LES COMPOSANTES AVANT LA DISCRIMINANTE IL FAUT UTILISER L'OPTION NOPRINT CAR SINON VOUS AUREZ TOUS LES GRAPHIQUES 2 A 2 

DEUXIEME ACM AVEC NOPRINT ET DIM= LE NOMBRE TOTAL DE DIMENSION SELON VOTRE FICHIER DE DONNEES ;
* les resultats sont dans le fichier qui s'appelle sortie; 

proc corresp data = fichier_donnees binary dim=NOMBRE_TOT_DE_DIM outc=sortie  ;
tables LISTE DE TOUTES VARIABLES Y COMPRIS  var_qual_a_expliquer;
supplementary var_qual_a_expliquer;
run;


proc corresp data = fichier_donnees mca dim=2 outc=sortie ;
tables LISTE DE TOUTES VARIABLES Y COMPRIS  var_qual_a_expliquer;
supplementary var_qual_a_expliquer;
run;

ods graphics off;



*SORTIE CONTIENT  DES INFORMATIONS AUTRES QUE SUR LES INDIVIDUS ON LES SUPPRIME ET ON GARDE LES COORDONNEES DE 1 à NOMBRE_TOT_DE_DIM  (ICI=24);
data resu_acm; set sortie;
keep dim1-dim24;
if _TYPE_ ='OBS';
run;

*IL FAUT AJOUTER LA var_qual_a_expliquer POUR AVOIR LE FICHIER ENTREE DE LA DISCRIM;
data cible;
set fichier_donnees;
keep var_qual_a_expliquer;
run;

data fich_tot;
merge cible resu_acm;
run;

*Analyse factoriellle discriminante sur variable qualitative AFD DISQUAL;
proc candisc data =fich_tot;
class var_qual_a_expliquer;
var dim1-dim24;
run;


* DISCRIMINATION BAYESIENNE ON CALCULE LE % DE MAL CLASSES PAR VALIDATION CROIS2E;
PROC DISCRIM DATA= FICH_TOT  all crossvalidate canonical;
class var_qual_a_expliquer;
var dim1-dim24;
run;

*DISCRIMINATION BAYESIENNE ON CALCULE LE % DE MAL CLASSES SUR UN ECHANTILLON TEST OBTENU AVEC LA PROC SURVEYSELECT;

*EXEMPLE IRIS DE FISHER;


proc surveyselect data =sashelp.iris

method =srs /*tirage aléatoire simple*/

n =40 /*il prend 40 observations de chaque groupe de taille 50 observations, donc on aura au total 120observations */

seed=530  /* on fixe la "graine" pour avoir à chaque réexécution le même échantillon sinon comme c'est un tirage aléatoire 
ça change à chaque fois*/

out=SampleStrata  /* crée un fichier qui s'appelle samplestrata */

outall; /* avec outall , le fichier samplestrata contient les données initiales et une variable selected qui prend 1 
si l'observation est choisi dans les 70% de l'échantillon de base et 0 sinon c'est à dire si 
l'observation est dans l'échant test surlequel il faut appliquer  la proc discrim pour valider le modèle*/

strata species; /* tirage stratifié, cela veut dire qu'on aura le même pourcentage d'observations de chaque groupe dans 
l'échantillon et dans la population*/
run ;
proc surveyselect data =sashelp.iris 

method=srs  /*idem que ci dessus*/

samprate=0.7  /*il prend 70% des observations de chaque groupe , voir le commentaire précédent sur strata*/
seed =520  /*idem que ci dessus*/
out =SampleStrata outall; /*idem que ci dessus*/
strata species; /*idem que ci dessus*/
run
;

;
proc print ; 
run;
/********netoyage des bases finales****************/
*création du fichier de base;
data base (keep=species SepalLength SepalWidth PetalLength PetalWidth);
set SampleStrata;
if selected=1;
run
; 
*création du fichier de test;
data test (keep=species SepalLength SepalWidth PetalLength PetalWidth);
set SampleStrata;
if selected=0;
run
; 
*creation des fichiers base et test en même temps;
data base (keep=species SepalLength SepalWidth PetalLength PetalWidth) test (keep=species SepalLength SepalWidth PetalLength PetalWidth);
set SampleStrata;
if selected=1 then output base;
else output test;
run
; 
*DISCRIMINANTE LINEAIRE HYPOTHESE DE NORMALITE;

/******** application de la discrim  avec echantillon test pour estimer sans biais le pourcentage de mal classés*****/

proc discrim data =base testdata =test  outstat =strateresult method =normal ; 
class Species;
/* c'est la variable à expliquer y */

var SepalLength SepalWidth PetalLength PetalWidth ;
/* les variables explicatives xi*/
run;

/******** application de la discrim  avec validation croisée pour estimer sans biais le pourcentage de mal classés*****/
proc discrim data =sashelp.iris CROSSVALIDATE   outstat =strateresult method =normal ; 
class Species;
/* c'est la variable à expliquer y */

var SepalLength SepalWidth PetalLength PetalWidth ;
/* les variables explicatives xi*/
run;


*DISCRIMINANTE LINEAIRE SANS HYPOTHESE DE NORMALITE    METHODES NON PARAMETRIQUES;

*analyse discrimante  décisionnelle méthode explicative ici on test les méthodes non paramétrique LES PLUS PROCHE VOISINS;
 
proc discrim data=FICH_TOT method = npar k=5 crossvalidate ; 
var DIM1-DIM24 ; 
class var_qual_a_expliquer;
run; 
*analyse discrimante  décisionnelle méthode explicative ici on test les méthodes non paramétrique DISCRIMINATION PAR 
BOULE DE RAYON R ;
 
proc discrim data=FICH_TOT method = npar R=0.14 crossvalidate ; 
var DIM1-DIM24 ; 
class var_qual_a_expliquer;
run;

* VOUS POUVEZ ESSAYER PLUSIEURS VALEURS DES PARAMETRES K POUR LES PPV OU R POUR LA DISCRIM PAR BOULE ET COMPARER LES 
DIFFERENTES METHODES EN TERMES DE % DE MAL CLASSES;

*SELECTION DES FACTEURS PAS A PAS;

PROC STEPDISC DATA= FICH_TOT  fw; * OPTION ASCENDANTE FOWARD;
class var_qual_a_expliquer;
var dim1-dim24;
run;


PROC STEPDISC DATA= FICH_TOT  bw;  * OPTION DESCENDANTE BACKWARD;
class var_qual_a_expliquer;
var dim1-dim24;
run;


PROC STEPDISC DATA= FICH_TOT  sw;* OPTION STEPWISE QUI FAIT UEN SORTE DE "COMBINAISON" DES 2 PRECEDENTES;
class var_qual_a_expliquer;
var dim1-dim24;
run;

*POUR CHAQUE METHODE VOUS AUREZ LES VARIABLES SELCTIONNEES VOUS REFEREZ ALORS PROC DISCRIM AVEC SEULEMENT LES VARIABLES SELECTIONNEES;

**************************************************************************************************************************
* CLASSIFICATION SUR LES COORDONNEES FACTORIELLES DE L'ACM;

ods graphics on;
proc cluster data=fich_tot method=ward out=ARBRE;
var DIM1-DIM24;
run;
*TRACE LE DENDROGRAMME ON MET COPY LES DIM POUR POUVOIR FAIRE UNE REPRESENTATION GRAPHIQUE DES CLASSES SUR LE PLAN FACTORIEL;
proc tree data=ARBRE out=PLAN nclusters=5 ;
COPY DIM1 DIM2;
run;
proc sgplot data=PLAN;
   scatter y=dim2 x=dim1 / group=cluster;
run;
ods graphics off;

