# Crop and weed segmentation

> Boidin Benoît, Casanova Aurélie, Lebreton Sébastien.

L'objectif de ce projet est d'entraîner un modèle de segmentation d'image pour détecter les cultures et les mauvaises herbes, à partir d'images aériennes.

## Dataset

### Prise en main

Avant d'entraîner le modèle, nous avons pris en main le fichier `dataset.py` permettant de charger le les données. Nous avons remarqué que les images étaient de haute qualité, et que les masques étaient noirs. Après analyse, les masques apparaissaient noirs mais présentaient des pixels avec des valeurs de 0 à 4, en cohérence avec le programme mentionné ci-dessus :

```python
    id2cls: dict = {0: "background",
                    1: "crop",
                    2: "weed",
                    3: "partial-crop",
                    4: "partial-weed"}
```

### Data augmentation

Afin d'améliorer les performances du classifieur, nous avons choisi la data augmentation. Voici la fonction que nous avons implémentée, dans le fichier `dataset.py` :

```python
    def apply_augmentation(self, input_img, target):
        augmentation_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        input_img = augmentation_pipeline(input_img)
        target = augmentation_pipeline(target)

        return input_img, target
```

Cette fonction permet d'appliquer aléatoirement une rotation horizontale ou verticale à l'image et au masque. Nous avons choisi ces deux rotations car elles sont les plus pertinentes pour notre problème. En effet, les images sont prises d'avion, et les cultures et mauvaises herbes sont donc vues de haut. Une rotation horizontale ou verticale ne change donc pas la nature de l'image.

Cependant, après avoir essayé d'entraîner le modèle, nous avons remarqué une augmentation de la loss, de 0.39 à 0.50.

## Modèle

Nous avons premièrement mis en place un modèle simple pour vérifier que tout fonctionnait correctement :

```python
model = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), 
                            torch.nn.Conv2d(64, dataset.get_class_number(), 3, padding=1))
```

Pour l'entraînement, nous avons utilisé le code fourni dans le projet :

```python
trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=cuda)
trainer.fit(train_loader, epoch=epoch)
```

Après quelques itérations, nous avons trouvé un modèle plus performant, qui nous a permis de passer de 1.5 à 0.8 de loss : 

```python
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Dropout2d(0.2),
                            torch.nn.Conv2d(64, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Dropout2d(0.2),
                            torch.nn.Conv2d(64, dataset.get_class_number(), 3, padding=1))
```

Les couches `BatchNorm2d` permettent de normaliser les données en entrée, et les couches `Dropout2d` permettent de réduire le surapprentissage. Le temps d'entraînement est significativement augmenté, mais les résultats sont meilleurs.

## Entraînement

Lors du premier entraînement, nous avons remarqué que le modèle convergeait, mais que le temps de calcul était très long. En effet, une époque sur nos machines personnelles prenait au moins 30 minutes.

Pour éviter ce problème, nous avons sélectionné quatre solutions :

- utiliser les machines du CREMI ;
- modifier l'entrainement pour utiliser le GPU de nos machines personnelles ;
- réduire le dataset pour les essais ;
- enregistrer le modèle à la fin d'un entraînement, et le charger pour la suite.

### Utilisation des machines du CREMI

L'utilisation des machines du CREMI nous a permis de réduire le temps d'entraînement à 10 minutes par époque. Nous avons donc pu faire plusieurs essais en utilisant une connexion SSH.

### Utilisation du GPU

Pour l'entrapinement sur GPU, nous n'avons pas pu utiliser CUDA, car nous ne disposons pas de carte graphique NVIDIA. Aussi, les machines du CREMI disposent d'une version de torch qui n'est pas compilée pour tirer parti du GPU. Après avoir passé du temps sur l'adaptation au processeurs graphiques ARM (Apple Silicon), nous avons abandonné car certains opérations n'étaient pas supportées sur MPS :

```shell
NotImplementedError: The operator 'aten::random_' is not currently implemented for the MPS device.
```

### Réduction du dataset

Nous avons hésité entre réduire le nombre de photos ou diminuer leur qualité, mais l'enregistrement du modèle nous a permis de rapidement tester notre code, et nous avons donc choisi cette solution.

### Enregistrement du modèle

Nous avons donc choisi l'enregistrement du modèle, ce qui nous a permit de rapidement tester notre code, en particulier pour afficher les résultats :

```python
    model.load_state_dict(torch.load(model_path))
    ...
    torch.save(model.state_dict(), "model.pth") 
```

### Paramètres

Nous avons commencé avec un `batch_size` de 32, ce qui signifie que 32 images sont traitées en même temps. Cependant, nous avons remarqué une pression importante sur la mémoire.

Au départ, nous avons confondu la taille du batch et le nombre de batch. Nous nous en sommes rendus compte lorsque nous avons essayé d'augmenter la taille du batch à 64, et que la mémoire était toujours pleine.
D'un autre côté, nous avons obtenu les résultats suivants :

| batch_size | loss |
|------------|------|
| 32         | 0.60 |
| 64         | 0.39 |

Avec un `batch_size` de 5, nous avons réussi à réduire le temps d'entraînement de 57%, car la ram n'était plus pleine, et la machine n'avait plus besoin d'utiliser de swap. La valeur de la loss est également descendue à 0.37, après seulement une époque.

---

Prochains entraînements :

- batch_size = 10
- epoch = 5

## Résultats

L'augmentation du nombre d'époque n'a pas permis d'augmenter l'efficacité du modèle.
Avec un `batch_size` de 5 et une époque, nous avons obtenu une loss de 0.37.
