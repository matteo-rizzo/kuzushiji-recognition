# Promemoria per Alessandro

## Darkflow

### Installazione

Per effettuare il training della rete, è necessario installare la libreria Darkflow, che fa da wrapper a Darknet estendendone la compatibilità a Python. Per effettuare correttamente l'installazione, segui la guida "darkflow_guide.md" o riferisciti direttamente alla [repo ufficiale](https://github.com/thtrieu/darkflow).

Dopo aver installato il pacchetto in Pycharm, è necessario creare il percorso ```libs/darkflow``` dalla root del progetto e installare anche lì il pacchetto con il comando ```pip install -e .```.

### Configurazione

Per utilizzare Darkflow, è necessario inizializzare il modello con dei pesi e con una configurazione specificati nel file ```networks/configuration/params_model_YOLO.json``` come file ```.weights``` e ```.cfg``` rispettivamente. Tali file devono essere omonimi.

Per pesi e configurazioni è necessario creare i seguenti percorsi:

* **Cartella dei pesi**: ```libs/darkflow/bin```;
* **Cartella delle configurazioni**: ```libs/darkflow/cfg```.

Perché i checkpoint funzionino correttamente è inoltre necessario creare una cartella ```ckpt``` nella root delle progetto. 

## Struttura del dataset

Darkflow richiede un'annotazione in XML secondo modello PASCAL VOC per ciascuna immagine. Di conseguenza, è necessario creare i seguenti percorsi:

* **Dataset Kaggle**: ```datasets/kaggle```, contenente i file:
  
  * ```classes.csv``` (file ```unicode_translation.csv```);
  * ```image_labels_map.csv``` (file ```train.csv```).
  
* **Dataset di training:** ```datasets/kaggle/training```
  
  * **Immagini di training: **```datasets/kaggle/training/images```
  * **Annotazioni di training:** ```datasets/kaggle/training/annotations```
  * **Backup delle immagini di training:** ```datasets/kaggle/training/backup```
  
* **Dataset di testing: **```datasets/kaggle/testing```
  
  * **Immagini di testing:** ```datasets/kaggle/testing/images```
  
* **Cartella delle etichette:** ```datasets/kaggle/labels```, contenente i file:

  * ```labels.txt``` 
  * ```class_name_to_number.csv```

  **Nota**: entrambi i file sono generati dallo script ```data_format_conversion```.

**Nota**: il backup delle immagini di training serve a ripristinare velocemente il dataset iniziale dopo averlo processato per renderlo compatibile con Darkflow.

## Script

Sono stati realizzati 4 pacchetti di script in ```scripts/```:

1. ```data_analysis```: esegue analisi e visualizzazione di base dei dati, ispirata ad un [kernel su Kaggle](https://www.kaggle.com/christianwallenwein/visualization-yolo-labels-useful-functions);

2. ```data_format_conversion```:
   1. Genera la mappatura delle etichette delle classi ad interi;
   2. Genera le annotazioni delle immagini del dataset in XML;
   3. Rinomina i file trasformano i caratteri "-" in "_" (per compatibilità con Darkflow).
   
   **Nota**: perché la generazione delle annotazioni abbia successo, è necessario apportare una piccola modifica al costruttore della classe ```Writer``` del pacchetto ```pascal_voc_writer```. In particolare, è necessario aggiungere un parametro ```filename``` da passare al parametro omonimo nella definizione del dizionario ```template_parameters``` (che si trova sempre nel costruttore).
   
3. ```dataset_resizing```: 

   1. Ridimensiona il dataset secondo una data dimensione;
   2. Ripristina il dataset iniziale a partire dal backup.
4. ```test_bounding_boxes```: data un'immagine di training, visualizza le bounding box relative alla sua annotazione. 

Per avviare correttamente gli script, è necessario creare delle nuove configurazioni in Pycharm.

**Nota**: è importante che la directory corrente della configurazione coincida con la root del progetto.



