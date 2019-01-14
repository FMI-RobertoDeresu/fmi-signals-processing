# FMI - Master IA an 2 - Prelucrarea semnalelor si aplicatii

## Proiect 1

Sa se implementeze algoritmii HMM:
1. Algoritmul de generare de observatii;
2. Algoritmul de evaluare forward;
3. Algoritmul de decodare Viterbi;
4. Algoritmul de invatare Baum-Welch. 

#### Vom considera cazul unidimensional, in care functiile de densitate de iesire sunt mixturi de doua Gaussiene.

La punctul 1, modelul HM este complet specificat (parametrii sunt cititi dintr-un fisier) si se genereaza secvente de observatii (T=20; 30).

La punctele 2 si 3, se aleg trei modele HM diferite, complet specificate. Unul dintre modele este folosit pentru generare de observatii. Secventele generate sunt apoi evaluate/decodate pe fiecare dintre cele 3 modele. La pasul 2 se afiseaza probabilitatea ca HMM sa fi generat secventa de observatii. La pasul 3 se afiseaza probabilitatea celei mai probabile secvente de stari care a produs secventa de observatii, precum si cea mai probabila secventa de stari.

La punctul 4, se da un model HM complet specificat si se genereaza secvente de observatii. Apoi parametrii modelului se perturbeaza (putin), iar observatiile generate anterior se folosesc in antrenarea modelului perturbat. Se aplica algoritmul de evaluare pe modelul initial, pe modelul perturbat si pe modelul antrenat si se compara rezultatele. Se compara parametrii initiali ai modelului HM cu parametrii obtinuti dupa antrenare.
