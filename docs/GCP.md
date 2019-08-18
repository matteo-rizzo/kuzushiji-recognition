# Uso di Google Cloud Platform (GCP)

## Base

```bash
gcloud auth login		# autenticati nel browser
```

```bash
gcloud projects list 	# lista progetti
```

```bash
gcloud config set project cognitive-services-248116		# seleziona un progetto
```

## Trasferire file a VM con SCP

```bash
gcloud compute scp file.txt instance_name:path_for_file
```

La nostra istanza si chiama: `tensorflow-gpu-p4-vm`.

### Esempi

```bash
gcloud compute scp file.txt tensorflow-gpu-p4-vm:~ # Mette nella home della VM
```

```bash
gcloud compute scp --recurse Folder/ tensorflow-gpu-p4-vm:~ # Copia la cartella Folder nella home della VM
```

## Connettersi in SSH

```bash
gcloud compute ssh instance_name
```

Nota: la prima volta chiede una passphrase. Se ti riconnetti ti chiede di reinserirla quindi ricordatela.

### Esempi

```bash
gcloud compute ssh tensorflow-gpu-p4-vm
```

## Trasferire file da VM a locale

```bash
gcloud compute scp instance_name:path_to_remote_file local_path_for_file
```

## Montare e smontare gc storage in file system VM

Monta il bucket come una cartella dentro la cartella `repo_project_cs/`.

```bash
gcsfuse --implicit-dirs -o nonempty dataset-cs repo_project_cs/
```

Per smontare la stessa cartella:

```bash
sudo umount repo_project_cs/
```

Se dice che la cartella è in uso provare ad aggiungere le flag `-l` e poi `-f`.

## Copiare tutto il dataset da gcloud storage alla VM

In alternativa ad usare storage come un file system remoto usando `gcsfuse` si può invece spostare il dataset sul disco della macchina VM.

```bash
gsutil -m rsync -r gs://dataset-cs/datasets path/to/datasets/folder/
```

Questo comando sincronizza la cartella di destinazione con il contenuto della cartella `datasets` nel bucket.

[Link docs](https://cloud.google.com/storage/docs/gsutil/commands/rsync#be-careful-when-synchronizing-over-os-specific-file-types-symlinks-devices-etc)

## Utility

Questo non dovrebbe servire, ma lo riporto comunque.

Se dovesse servire reinstallare i driver NVIDIA sulla VM usare il seguente comando dalla console della VM:

```bash
sudo /opt/deeplearning/install-driver.sh
```
