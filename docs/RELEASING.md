# Publier une version de VoiceBridge

Les releases sont automatiques : le workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
incrémente la version avec [`scripts/bump_version.py`](../scripts/bump_version.py), crée le tag
`vX.Y.Z`, compile les installeurs et publie la release GitHub avec `SHA256SUMS.txt`.

| Dans le dernier message de commit | Effet                     |
| --------------------------------- | ------------------------- |
| *(rien)*                          | patch : 0.3.1 → 0.3.2     |
| `[minor]`                         | mineure : 0.3.1 → 0.4.0   |
| `[major]`                         | majeure : 0.3.1 → 1.0.0   |
| `[skip release]`                  | aucune release            |

## Notes de release

Le texte de chaque release vient de [`packaging/RELEASE_NOTES.md`](../packaging/RELEASE_NOTES.md),
suivi de la liste des changements générée par GitHub. Il est réutilisé à chaque version : il doit
rester **court, en anglais** et ne pas décrire une version précise.

## À savoir

- **Chaque push sur `main` publie une vraie release**, y compris le premier push des changements
  qui ont introduit ce mécanisme. Le numéro augmente une fois par push, pas une fois par commit.
- **La CI ajoute un commit sur `main` après chaque push** (`chore(release): vX.Y.Z [skip ci]`).
  Faites `git pull --rebase` avant de travailler à nouveau, sinon votre prochain push sera refusé.
- **Si `main` est protégée** et interdit les pushs directs, l'étape `version` échouera : il faut
  autoriser GitHub Actions à pousser sur la branche (règle de protection → bypass pour
  `github-actions[bot]`).
- **Si une plateforme échoue à la compilation**, le tag est créé mais aucune release n'est
  publiée. Le push suivant crée simplement la version d'après.
- **Les personnes déjà en v0.3.0 n'ont pas le bouton de mise à jour** : elles doivent installer
  la v0.3.1 à la main une fois. Ensuite, l'application propose elle-même les mises à jour.

## Vérifier une modification de la CI sans publier

Ouvrez une pull request : la CI compile et teste toutes les plateformes, mais l'étape `version`
et la release ne s'exécutent que sur `main`.
