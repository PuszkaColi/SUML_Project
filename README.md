# Przewidywanie cukrzycy

## Opis apliakcji

Apliakcja dotyczy prognozowania występowania cukrzycy u danej osoby, biorąc pod uwagę
wiele czynników takich jak:

- **Wiek**: Wiek jest ważnym czynnikiem w przewidywaniu ryzyka cukrzycy. Wraz z wiekiem ryzyko zachorowania na cukrzycę wzrasta. Wynika to częściowo z takich czynników, jak zmniejszona aktywność fizyczna, zmiany poziomu hormonów i większe prawdopodobieństwo wystąpienia innych schorzeń, które mogą przyczyniać się do cukrzycy.
 
- **Płeć**: Płeć może odgrywać rolę w ryzyku cukrzycy, chociaż efekt może być różny. Na przykład kobiety z historią cukrzycy ciążowej (cukrzycy w czasie ciąży) mają większe ryzyko zachorowania na cukrzycę typu 2 w późniejszym życiu. Ponadto niektóre badania sugerują, że mężczyźni mogą mieć nieco wyższe ryzyko zachorowania na cukrzycę w porównaniu z kobietami.
 
- **Wskaźnik masy ciała**: (BMI): BMI to miara tkanki tłuszczowej oparta na wzroście i wadze osoby. Jest powszechnie stosowany jako wskaźnik ogólnej masy ciała i może być pomocny w przewidywaniu ryzyka cukrzycy. Wyższy wskaźnik BMI wiąże się z większym prawdopodobieństwem zachorowania na cukrzycę typu 2. Nadmiar tkanki tłuszczowej, szczególnie w okolicy talii, może prowadzić do insulinooporności i upośledzać zdolność organizmu do regulowania poziomu cukru we krwi.
 
- **Nadciśnienie**: Nadciśnienie, czyli wysokie ciśnienie krwi, to stan, który często współwystępuje z cukrzycą. Te dwa stany mają wspólne czynniki ryzyka i mogą przyczyniać się do rozwoju drugiego. Nadciśnienie zwiększa ryzyko rozwoju cukrzycy typu 2 i odwrotnie. Oba stany mogą mieć szkodliwy wpływ na zdrowie układu krążenia.
 
- **Choroba serca**:Choroby serca, w tym choroby takie jak choroba tętnic wieńcowych i niewydolność serca, są związane ze zwiększonym ryzykiem cukrzycy. Związek między chorobą serca a cukrzycą jest dwukierunkowy, co oznacza, że posiadanie jednego stanu zwiększa ryzyko rozwoju drugiego. Dzieje się tak, ponieważ mają wiele wspólnych czynników ryzyka, takich jak otyłość, wysokie ciśnienie krwi i wysoki poziom cholesterolu.
 
- **Historia palenia**: Palenie jest modyfikowalnym czynnikiem ryzyka cukrzycy. Stwierdzono, że palenie papierosów zwiększa ryzyko rozwoju cukrzycy typu 2. Palenie może przyczyniać się do insulinooporności i zaburzać metabolizm glukozy. Zaprzestanie palenia może znacznie zmniejszyć ryzyko rozwoju cukrzycy i jej powikłań.
 
- **Poziom HbA1c**: HbA1c (hemoglobina glikowana) to miara średniego poziomu glukozy we krwi w ciągu ostatnich 2-3 miesięcy. Dostarcza informacji o długoterminowej kontroli poziomu cukru we krwi. Wyższe poziomy HbA1c wskazują na gorszą kontrolę glikemii i są związane ze zwiększonym ryzykiem rozwoju cukrzycy i jej powikłań.
 
- **Poziom glukozy we krwi**: Poziom glukozy we krwi odnosi się do ilości glukozy (cukru) obecnej we krwi w danym momencie. Podwyższony poziom glukozy we krwi, szczególnie na czczo lub po spożyciu węglowodanów, może wskazywać na upośledzoną regulację glukozy i zwiększać ryzyko rozwoju cukrzycy. Regularne monitorowanie poziomu glukozy we krwi jest ważne w diagnozie i leczeniu cukrzycy.


# Instrukcja użytkowania

## Uruchomienie lokalnie

1. Sklonuj repozytorium projektu z Github:
    ```bash
    git clone https://github.com/PuszkaColi/SUML_Project.git
    ```

2. Z poziomu folderu projektu zainstaluj potrzebne biblioteki:
    ```bash
    pip install -r requirements.txt
    ```
3. Uruchom plik app.py:
    ```bash
    python3 app.py
    ```
   
4. W przeglądarce wejdź na adres: http://localhost:5000/

## Uruchomienie aplikacji z serwera w chmurze:
1. Wejdź w link: https://przewidywaniecukrzycy-dyb4a5g6b7b3hcg2.polandcentral-01.azurewebsites.net/
2. Jeśli powyższy link nie działa wejdź w ten: https://web-production-6249.up.railway.app/
