# Uso di GCP

## Base

gcloud auth login		# autenticati nel browser

gcloud project list 	# lista progetti

gcloud config set project cognitive-services-248116

## Trasferire file a VM con SCP

gcloud compute scp [file.txt][istanza:percorso-in-cui-mettere-il-file]

Esempi:

gcloud compute scp file.txt instance-4cpu-1gpu-p4:~ # Mette nella home della VM
gcloud compute scp --recurse Folder/ instance-4cpu-1gpu-p4:~ # Copia la cartella Folder nella home della VM

## Connettersi in SSH

gcloud compute ssh [nome-istanza]

La prima volta chiede una passphrase. Se ti riconnetti ti chiede di reinserirla quindi ricordatela.

Esempio:

gcloud compute ssh instance-4cpu-1gpu-p4

## Trasferire file da VM a locale (utile per i log file)

gcloud compute scp [istanza:percorso-in-cui-si-trova-il-file-remoto][percordo-file-dove-vuoi-metterlo]