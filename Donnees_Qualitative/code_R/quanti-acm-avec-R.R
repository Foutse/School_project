######################################################################################
##                                                                                  ##
##     Repr�senter graphiquement les r�sultats d'une analyse factorielle avec R     ##
##                                                                                  ##
##               (c) Anton Perdoncin, Pierre Merckl� et Quanti 2014                 ##
##                                                                                  ##
######################################################################################


### 1. REALISER UNE ACM AVEC FACTOMINER

## A. Installer et charger le package FactoMiner
library(FactoMineR)

## B. T�l�charger et pr�parer les donn�es

# Installer et charger le package questionr
library(questionr)

# Lire les donn�es et les stocker dans l'objet DONNEES
data(hdv2003)
SOURCE <- hdv2003
# Avoir un aper�u des donn�es
str(SOURCE)
# S�lectionner les variables retenues pour l'analyse et les stocker dans l'objet DATA
DONNEES <- subset(SOURCE, select=c("hard.rock", "lecture.bd", "peche.chasse", "cuisine", "bricol", "cinema", "sport", "age", "sexe", "qualif"))
# Recoder et factoriser la variable age
DONNEES$age <- cut(DONNEES$age, breaks=quantile(DONNEES$age))
# Avoir un aper�u des donn�es
str(DONNEES)

## C. R�aliser l'ACM et stocker les r�sultats

# Faire l'ACM et ranger les r�sultats dans l'objet resultats
resultats <- MCA(DONNEES, ncp=7, quali.sup=8:10, graph=FALSE)
# Voir les r�sultats
print(resultats)
# Avoir un aper�u des r�sultats
summary(resultats)



### 2. COMBIEN DE FACTEURS FAUT-IL RETENIR ? L'HISTOGRAMME DES VALEURS PROPRES

# Afficher les valeurs propres
print(resultats$eig)
# Repr�senter l'histogramme des valeurs propres
barplot(resultats$eig[, 2], main="Histogramme des valeurs propres", names.arg=rownames(resultats$eig), xlab="Axes", ylab="Pourcentage d'inertie", cex.axis=0.8, font.lab=3, col="orange")



### 3. ESSAYER LES FONCTIONNALITES GRAPHIQUES DU PACKAGE FACTOMINER

# Repr�senter graphiquement les r�sultats en 4 graphiques

# D�finir la r�partition : 2x2
par(mfrow=c(2, 2))
# Variables actives et suppl�mentaires, avec le param�tre choix="var"
plot.MCA(resultats, choix="var", title="Variables actives et suppl�mentaires", axes=c(1, 2))
# Modalit�s actives
plot.MCA(resultats, invisible=c("ind", "quali.sup"), title="Modalit�s actives", axes=c(1, 2))
# Modalit�s suppl�mentaires
plot.MCA(resultats, invisible=c("ind", "var"), title="Modalit�s suppl�mentaires", axes=c(1, 2))
# Individus
plot.MCA(resultats, invisible=c("quali.sup", "var"), title="Nuage des individus", axes=c(1, 2))



### 4. REALISER SOI-MEME UN GRAPHIQUE SUR-MESURE

## A. Quelles modalit�s repr�senter ?

# Contributions des modalit�s actives aux facteurs 1 et 2
round(resultats$var$contrib[,1:2], 1)
# S�lectionner � la main les modalit�s qui contribuent le plus fortement aux facteurs 1 et 2
moda <- c(6, 9:14)
# Ou les s�lectionner automatiquement : garder celles dont les contributions sont sup�rieures au seuil...
seuil <- 100/nrow(resultats$var$contrib)
moda <- which(resultats$var$contrib[, 1]>seuil | resultats$var$contrib[, 2]>seuil)

# Visualiser les coordonn�es des modalit�s s�lectionn�es
round(resultats$var$coord[moda, 1:2], 2)

## B. Le graphique des modalit�s pas � pas


# Premiere �tape : cr�er un cadre graphique "vide"
windows(10, 7)
par(mfrow=c(1,1))
plot(resultats$var$coord[moda, 1:2]*1.2, type="n", xlab=paste0("Axe 1 (", round(resultats$eig[1,2], 1), "%)"), ylab=paste0("Axe 2 (", round(resultats$eig[2,2], 1), "%)"), main="Premier plan factoriel", cex.main=1, cex.axis=0.8, cex.lab=0.7, font.lab=3)
abline(h=0, v=0, col="grey", lty=3, lwd=1)

# Seconde �tape : tracer les points des fortes contributions
points(resultats$var$coord[moda, 1:2], col="black", pch=c(15, 16, 16, 17, 17, 6, 6), cex=1.5)

# Troisieme etape : ajouter les libell�s
etiquettes <- rownames(resultats$var$coord)
print(etiquettes)
text(resultats$var$coord[moda,1:2], labels=etiquettes[moda], col="black", cex=1, pos=4)

# Quatri�me �tape : Ajouter les modalit�s suppl�mentaires
print(resultats$quali.sup$coord[, 1:2])
text(resultats$quali.sup$coord[c(1:4, 6:7, 9:13), 1:2]*1.2, labels=rownames(resultats$quali.sup$coord[c(1:4, 6:7, 9:13), ]), cex=0.8, col="blue", font=3)

# Cinqui�me �tape : Relier des points entre eux : tranches d'�ge
lines(resultats$quali.sup$coord[1:4,1:2], col="blue", lty=3)

# Sixi�me �tape : Ajouter une l�gende
legend("topleft", legend=c("P�che et chasse", "Bricolage", "Cin�ma", "Sport", "Suppl�mentaires"), bty="y", bg="white", text.col=c(1,1,1,1,"blue"), col=c(1,1,1,1,"blue"), pch=c(15,16,17,6,0), cex=0.8, pt.cex=c(1,1,1,1,0))

## C. Le nuage des individus pas � pas

# a. Repr�senter le nuage des individus
windows(10, 7)
par(mfrow=c(1,1))
plot(resultats$ind$coord[, 1:2], type="n", xlab=paste0("Axe 1 (", round(resultats$eig[1,2], 1), "%)"), ylab=paste0("Axe 2 (", round(resultats$eig[2,2], 1), "%)"), main="Nuage des individus", cex.main=1, cex.axis=0.8, cex.lab=0.8, font.lab=3)
abline(h=0, v=0, col="grey", lty=3, lwd=1)
points(resultats$ind$coord[, 1:2], col = rgb(0, 0, 0, 0.1), pch = 19)

# b. Habiller Le nuage des individus en fonction du sexe
windows(10, 7)
par(mfrow=c(1,1))
plot(resultats$ind$coord[, 1:2], type="n", xlab=paste0("Axe 1 (", round(resultats$eig[1,2], 1), "%)"), ylab=paste0("Axe 2 (", round(resultats$eig[2,2], 1), "%)"), main="Nuage des individus selon le sexe", cex.main=1, cex.axis=0.8, cex.lab=0.8, font.lab=3)
abline(h=0, v=0, col="grey", lty=3, lwd=1)
points(resultats$ind$coord[,1:2], col=as.numeric(DONNEES$sexe), pch=19)
legend("topright", legend=levels(DONNEES$sexe), bty="o", text.col=1:2, col=1:2, pch=19, cex=0.8)

# c. Pond�rer le nuage des individus

# Calculer le tableau du nombre d'individus par position selon les modalit�s d'une variable split
# Choisir la variable split
split <- DONNEES$qualif
# Calculer les tableaux du nombre d'invididus par position pour chaque modalit� du split
t <- by(resultats$ind$coord[, 1:2], split, table)
# Transformer les tableaux en dataframes
t <- lapply(t, as.data.frame, stringsAsFactors=FALSE)
# Fusionner les tableaux dans un seul tableau avec autant de colonnes que de modalit�s de split, contenant le nombre d'individus par position pour chaque modalit� du split
tableau <- data.frame(list(Dim.1="", Dim.2=""))[-1,]
for (i in 1:length(t)) {tableau <- merge(tableau, t[[i]], by=c("Dim.1", "Dim.2"), suffixes=paste0(".", (i-1):i), all=TRUE)}
# Ajouter au tableau une variable size calculant le nombre total d'individus par position
tableau$size <- apply(tableau[, -(1:2)], 1, sum, na.rm=TRUE)
# Ne garder que les positions sur lesquelles le nombre d'individus est sup�rieur � 0
tableau <- tableau[tableau$size>0, ]
# Chercher ligne par ligne la modalit� qui a le plus d'effectifs sur chaque position
tableau$col <- apply(tableau[,3:(2+length(t))], 1, function(x){which(x==max(x, na.rm=TRUE))[1]})

# Repr�senter le nuage pond�r�
windows(10, 7)
par(mfrow=c(1,1))
plot(tableau[, 1:2], type="n", xlab=paste0("Axe 1 (", round(resultats$eig[1,2], 1), "%)"), ylab=paste0("Axe 2 (", round(resultats$eig[2,2], 1), "%)"), main="Nuage pond�r� des individus", cex.main=1, cex.axis=0.8, cex.lab=0.8, font.lab=3)
abline(h=0, v=0, col="grey", lty=3, lwd=1) # lignes horizontales et verticales
# Repr�senter le nuage des individus
points(tableau[, 1:2], pch=19, cex=1+tableau$size*10/max(tableau$size), col=tableau$col)
legend("topright", legend=levels(split), bty="o", text.col=1:length(t), col=1:length(t), pch=20, pt.cex=1.5, inset=-0.01, y.intersp=0.7, cex=0.7)
