# dsp_nilm
Non intrusive monitoring DSP algorithm for electrical loads activation analysis - Engineering Electronic Degree thesis

Il presente lavoro di tesi si pone l’obiettivo di definire e implementare un algoritmo di disaggregazione dei carici elettrici (Non Intrusive Load Monitoring) con precise carat- teristiche: real-time ed event based. Si è ricercato un metodo in grado di identificare le principali appliances tipiche di un abitazione domestica tramite un approccio DSP, quindi con ridotte risorse computazionali. Lo studio ed i test effettuati sono basati sul dataset UK-dale, utilizzando un approccio decisionale che utilizza la distanza euclidea rispetto ad un database creato ad hoc. La supervisione, in confronto agli algoritmi che sfruttano tecniche di Machine Learning, è minima e i risultati ad essi comparabili. Il linguaggio di programmazione scelto è il Python 2.7.

VERSIONE 0 - Il codice caricato corrisponde a quello utilizzato nel lavoro di tesi. Per eseguirlo utilizzare il file principale "MAIN_SCRIPT", il dataset aggregato e disaggregato è contenuto nelle cartelle "test"  e "train". E' già salvata una copia di result che può essere visualizzata senza eseguire l'intero script. 
Il codice è abbondatemente commentato e suddiviso in file secondo le varie parti dell'algoritmo. A breve intendo caricare una nuova versione: allegando con una piccola guida sull'utilizzo dei vari file, incrementando i commenti e le spiegazioni delle varie parti di codice, così da facilitarne la comprensione e l'utilizzo.

Good luck! ;)
                                                                                                              Rodolfo Saraceni
                                                                                                            rodo.sara@gmail.com
