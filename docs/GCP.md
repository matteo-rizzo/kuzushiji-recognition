# Uso di Google Cloud Platform (GCP)

## Base

```bash
gcloud auth login		# autenticati nel browser
```

```bash
gcloud project list 	# lista progetti
```

```bash
gcloud config set project cognitive-services-248116		# seleziona un progetto
```

## Trasferire file a VM con SCP

```bash
gcloud compute scp file.txt instance_name:path_for_file
```

### Esempi

```bash
gcloud compute scp file.txt instance-4cpu-1gpu-p4:~ # Mette nella home della VM
```

```bash
gcloud compute scp --recurse Folder/ instance-4cpu-1gpu-p4:~ # Copia la cartella Folder nella home della VM
```

## Connettersi in SSH

```bash
gcloud compute ssh instance_name
```

Nota: la prima volta chiede una passphrase. Se ti riconnetti ti chiede di reinserirla quindi ricordatela.

### Esempi

```bash
gcloud compute ssh instance-4cpu-1gpu-p4
```

## Trasferire file da VM a locale

```bash
gcloud compute scp instance_name:path_to_remote_file local_path_for_file
```