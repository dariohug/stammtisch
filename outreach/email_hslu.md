**An:** thomas.koller@hslu.ch
**Betreff:** Anfrage Swisslos-Jass-Daten (DL4G) für ein privates KI-Projekt + kleiner Fund in jass-kit

---

Grüezi Herr Koller

Mein Name ist Dario Hug. In meiner Freizeit entwickle ich «Stammtisch», eine KI für den Schieber-Jass. Ihr `jass-kit` und das Material aus dem DL4G-Modul waren dafür ein sehr hilfreicher Ausgangspunkt, danke dafür!

Ich habe gesehen, dass die Swisslos-Spielprotokolle, die Sie für DL4G aufbereitet haben (ca. 1.8 Mio. Spiele, Oktober 2017 bis April 2018, sowie `2018_10_18_trump.csv`), in mehreren öffentlichen GitHub-Repositories von Studierenden liegen. Bevor ich sie weiter verwende, möchte ich Sie direkt fragen:

1. **Nutzung:** Darf ich die Daten für ein nicht-kommerzielles Projekt verwenden, also zum Trainieren von Modellen und für Auswertungen? Die Resultate und meinen Code würde ich veröffentlichen, die Rohdaten nicht weitergeben. Gibt es Bedingungen seitens Swisslos, die ich beachten muss?
2. **Weitere Daten:** Gibt es neuere oder umfangreichere Daten, z. B. mit Weis/Stöck, ganzen Partien oder Angaben zur Spielstärke?
3. **Zitieren:** Wie soll ich die Datenquelle korrekt angeben?

Als kleines Dankeschön ein Fund aus meinen Tests: Ich habe meine Engine gegen `RuleSchieber.get_valid_cards` und alle 1.82 Mio. Swisslos-Spiele geprüft. Liegen bereits zwei Trümpfe im Stich, bestimmt `jass-kit` den höchsten davon über den Kartenindex statt über die Trumpf-Rangfolge (`lowest_trump_played < current_trick[2]`). Dadurch wird in rund 0.15 % der zufälligen Spielsituationen Untertrumpfen erlaubt.

*Beispiel:* Trumpf Rosen, Stich SK – HA – H10, Hand {HK, D7}. `jass-kit` erlaubt HK, obwohl HK unter dem HA liegt. Regelkonform ist nur D7. Die Swisslos-Daten selbst enthalten keinen solchen Zug. Gerne schicke ich Ihnen den Testfall oder einen Pull Request.

Ich würde mich sehr über eine kurze Rückmeldung freuen.

Freundliche Grüsse
Dario Hug
d.hugwagner@gmail.com
