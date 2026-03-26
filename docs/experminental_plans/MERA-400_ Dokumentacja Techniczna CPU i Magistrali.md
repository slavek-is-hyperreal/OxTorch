# **MERA-400: Kompletna Dokumentacja Techniczna Architektury CPU i Magistrali**

**Przygotował:** Sławomir Majcher (slavek.majcher@gmail.com)  
**Tematyka:** Analiza techniczna asynchronicznego procesora oraz interfejsu (magistrali) polskiego minikomputera MERA-400.

## **Wstęp**

MERA-400 to unikalna konstrukcja w historii polskiej informatyki, wywodząca się z architektury K-202 (projektu Jacka Karpińskiego, Elżbiety i Andrzeja Jezierskich z 1971 roku). Główną cechą odróżniającą ten system od wielu współczesnych mu i dzisiejszych architektur jest głęboka asynchroniczność – zarówno na poziomie centralnej jednostki obliczeniowej (CPU), jak i magistrali systemowej. *Nota bene*, podejście to pozwalało ominąć ograniczenia wynikające z nierównej jakości komponentów elektronicznych dostępnych w krajach bloku wschodniego.  
Przejdźmy *ad rem* i przyjrzyjmy się szczegółowo technicznej budowie obu tych kluczowych podsystemów.

## **CZĘŚĆ I: ASYNCHRONICZNY PROCESOR (CPU)**

Procesor Mery 400 nie jest pojedynczym układem scalonym. *De facto*, jest to zespół sześciu płytek drukowanych (pakietów) o wymiarach 30x30 cm, zainstalowanych w jednostce centralnej na pozycjach od 4 do 9\.

### **1\. Budowa Fizyczna**

Całe CPU składa się z około 530 układów scalonych, co przekłada się na ekwiwalent około 20 000 tranzystorów (wpisując się idealnie w Prawo Moore'a dla roku 1976 w zestawieniu np. z procesorem Intel 8086).  
Użyto tu układów logicznych TTL z bipolarnej serii 74 (w newralgicznych miejscach szybszej serii 74H). Głównie są to układy rodzimej produkcji CEMI, ale również Tesla, Tungsram, a w rzadszych przypadkach (gdy brakowało wschodnich odpowiedników) układy zachodnie.  
Procesor dzieli się na 6 pakietów: trzem z nich przypisano przetwarzanie danych, a trzem sterowanie procesorem.

1. **PA (Pakiet Arytmometru)** \- główny blok obliczeniowy. Zawiera jednostkę arytmetyczno-logiczną (ALU), rejestry AC i AT, główne szyny wewnętrzne procesora (szyna A, szyna W), rejestr adresowy oraz licznik rozkazów.  
2. **PR (Pakiet Rejestrów)** \- blok rejestrów uniwersalnych, Rejestr R0 (przechowujący m.in. flagi procesora), rejestr ładowania binarnego oraz magistrale L i KI.  
3. **PP (Pakiet Przerwań)** \- układ logiki przerwań i ich priorytetów, rejestr zgłoszeń, rejestr przyjęć przerwań oraz rejestr statusu procesora SR (zawierający m.in. maski przerwań).  
4. **PX (Pakiet Sterowania)** \- zawiaduje głównym cyklem i stanami procesora (tu rezyduje automat skończony).  
5. **PM (Pakiet Mikrooperacji)** \- generuje fizyczne sygnały wyzwalające dla wszystkich układów na podstawie aktualnego stanu automatu.  
6. **PD (Pakiet Dekodera)** \- dekoduje pobrane z pamięci instrukcje.

### **2\. Architektura Sterowania: Automat Skończony**

Większość procesorów z tamtych i dzisiejszych lat bazuje na tzw. mikroprogramowaniu (gdzie wewnętrzny ROM zawiera sekwencje mikroinstrukcji dla każdego rozkazu). Zespół projektowy Mery 400 (w tym Elżbieta Jezierska) z powodu braku narzędzi (mikroasemblerów, sprzętu do programowania pamięci PROM) musiał wybrać inne rozwiązanie. *Sine qua non* dla powstania maszyny było zbudowanie sterowania w oparciu o sprzętowy **automat skończony (Finite State Machine)**, rozrysowywany ręcznie na wielkich płachtach papieru A1.

* Automat ten posiada 24 stany.  
* Składa się z głównych pętli: pętli pobrania rozkazu, pętli wykonania rozkazu, obsługi przerwań oraz pętli obsługi kluczy pulpitu.  
* Wejście w dany stan powoduje uruchomienie konkretnej, specyficznej dla niego grupy "mikrooperacji" (niektóre są bezwarunkowe, inne zależne od np. bitów danego rozkazu).

### **3\. Taktowanie Asynchroniczne**

W procesorze *expressis verbis* brakuje jednego, nieprzerwanie tykającego zegara (kwarcu). Jest on układem klasy **Self-Timed** (samotaktującym się) z elementami klasy **Speed-Independent** (niezależnymi od prędkości, np. podczas obsługi pamięci).  
Czas potrzebny na wykonanie mikrooperacji nie jest wyznaczony kwantem globalnego zegara, lecz precyzyjnymi elementami opóźniającymi ("timerami"), dostosowanymi do fizycznego czasu propagacji sygnału. Takie rozwiązanie pozwalało niwelować różnice w jakości komponentów od dostawców z bloku wschodniego – każdą maszynę po prostu "dostrajano".  
Do taktowania wykorzystuje się centralny układ na pakiecie PX generujący 7 sygnałów opartych o 5 elementów opóźniających (potencjometrów strojonych ręcznie na krawędzi pakietu):

* **STROB 1** – główny sygnał uruchamiający mikrooperacje w danym stanie. Jego długość zależy od konkretnego stanu (może przybierać jedną z 3 różnych długości w zależności od złożoności obliczeń).  
* **STROB 2** – sygnał o stałej długości, generowany tylnym zboczem sygnału STROB 1, doprowadzony tylko do tych stanów, które wymagają drugiego, sekwencyjnego kroku.  
* **GOT** – sygnał opóźniający używany do zapisania nowego stanu do rejestru automatu (taktuje przejścia między stanami).  
* **PC (Początek Cyklu)** i **KC (Koniec Cyklu)** – dwa dodatkowe sygnały (strojone na pakiecie PM) obsługujące specjalne momenty procesora.

Stan **P1** (Pobranie rozkazu z pamięci) jest unikalny – procesor deaktywuje w nim wszystkie swoje timery i zdaje się w 100% na zewnętrzną magistralę, wykorzystując protokół potwierdzeń (handshake). Cykl zostaje wznowiony dopiero w momencie asynchronicznej odpowiedzi z modułu pamięci.

## **CZĘŚĆ II: ASYNCHRONICZNA MAGISTRALA (INTERFEJS)**

Magistrala (Interfejs) systemu MERA-400 to *ex definitione* "zespół reguł i środków technicznych umożliwiających komunikację między poszczególnymi modułami". Pozwala ona na współistnienie na jednej szynie modułów starych (wolnych) i nowych (szybkich) bez żadnych zmian w taktowaniu.

### **1\. Zespół Reguł (Logika Magistrali)**

Komunikacja zawsze opiera się na relacji **Nadawca** (inicjator) – **Adresat** (odbiorca). Architektura definiuje 5 rodzajów transmisji:

1. **Zapis do pamięci (W)** – Nadawcą jest procesor/kanał DMA, adresatem blok pamięci.  
2. **Odczyt z pamięci (R)** – j.w.  
3. **Przesłanie (S)** – Wysłanie danych do kanału We/Wy.  
4. **Pobranie (F)** – Odczyt danych z kanału We/Wy.  
5. **Przerwanie (IN)** – Kanał We/Wy wysyła sygnał do CPU (lub jeden procesor do drugiego).

**Sygnały logiczne biorące udział w transmisji:**

* Sygnał wiodący pytania (np. **W**, **R**, **S**, **F**, **IN**).  
* **PN** (1 bit) – Numer procesora.  
* **NB** (4 bity) – Numer Bloku Pamięci (MERA-400 dzieli pamięć na 16 segmentów).  
* **AD** (16 bitów) – Adres komórki w bloku (lub adres urządzenia).  
* **DT** (16 bitów) – Dane.  
* **QB** – Wskaźnik określający czy operację zlecił System Operacyjny, czy program użytkownika.

**Protokół Handshake:**

1. Nadawca wystawia sygnały danych, adresu oraz sygnał wiodący pytania.  
2. Wszyscy użytkownicy magistrali sprawdzają sygnał. Jeden moduł rozpoznaje, że transmisja dotyczy jego adresu i staje się Adresatem.  
3. Adresat akceptuje dane (przy zapisie) lub wystawia dane (przy odczycie), a następnie potwierdza to sygnałem odpowiedzi **OK**.  
4. Nadawca rejestruje OK i ściąga swój sygnał wiodący pytania z magistrali.  
5. Adresat rejestruje koniec pytania i ściąga sygnał OK. Transmisja zostaje zakończona.

**Sytuacje wyjątkowe (Awaryjne):**

* **EN (Zajętość)** – Jeśli adresat istnieje, ale jest zajęty, zamiast OK odpowiada sygnałem EN. Zmusza to nadawcę (program) do późniejszego ponowienia operacji.  
* **PE (Błąd Parzystości)** – Aktywowany równolegle z OK przez pamięć, jeśli odczytano dane z uszkodzonym bitem parzystości.  
* **Brak odpowiedzi (Timeout)** – Jeśli wybrany adres nie istnieje, nadawca ma sprzętowy unifibrator (uzbrajany przy każdej transmisji na kilkanaście mikrosekund). Gdy czas upłynie, nadawca przerywa transmisję awaryjnie i generuje wyjątek (np. zapalenie lampki na pulpicie CPU).

### **2\. Zespół Środków Technicznych (Warstwa Fizyczna)**

Implementacja opiera się na tzw. **pakietach interfejsu (ISP lub ISK)** instalowanych w każdej szufladzie systemu. Wszystkie sygnały przed wyjściem lub wejściem do układów szuflady przechodzą przez ten pakiet.  
**Logika Odwrotna i Otwarty Kolektor:**

* Fizycznie wymuszenie dostępu jednej magistrali odbywa się przez ujemną logikę i bramki NAND z tzw. **otwartym kolektorem** (Open Collector), jako że w tamtych czasach niedostępne były układy trójstanowe (Tri-state).  
* W stanie spoczynkowym magistrala utrzymywana jest na wysokim napięciu poprzez rezystory podciągające. *Logiczne zero na magistrali odpowiada stanowi wysokiego napięcia*, a *logiczna jedynka to wymuszenie zwarcia do masy (niskie napięcie)*.  
* Pozwala to na zapięcie wielu nadajników na jednym przewodzie bez ryzyka zwarcia logicznego.

**Topologia fizyczna i Linie Długie:**

* Sygnały wyprowadzone do innych szuflad tworzą zjawisko elektroniki falowej ("Linii Długich").  
* Wyjściowe bramki sterujące szynami do innych szuflad to wzmocnione bramki 7438, pracujące po dwie równoleglekle (aby znieść prądy rzędu 60 mA).  
* Na pierwszym i ostatnim pakiecie w całym łańcuchu systemu instaluje się **Terminatory Interfejsu (TIF)** – pakiety z drabinkami rezystorowymi dopasowującymi impedancję falową kabli (niwelują odbicia sygnałów).

**Zabezpieczenia Zasilania (*Sui Generis* Izolacja):**

* **Sygnał OFF i ZF:** Jeśli zasilacz w szufladzie wykryje padające napięcie, czeka na zakończenie bieżącej transmisji na magistrali (sygnał ZF \- Zezwolenie na odłączenie), a następnie generuje sygnał OFF, który zatrzymuje sterowanie bramkami Open Collector (wprowadza je w stan wysokiej impedancji). Dzięki temu "umierająca" szuflada w ułamku sekundy sprzętowo odcina się od magistrali i nie powoduje śmieciowych transmisji paraliżujących resztę systemu.  
* **Zasilanie Rezerwowe (15V):** Układ rezerwacji interfejsu oraz terminatory TIF muszą mieć prąd bezwzględnie zawsze. Dlatego w kablu magistrali dystrybuowana jest specjalna, zabezpieczona diodami linia 15V. Na każdym pakiecie interfejsu lokalny regulator analogowy zbija to do 5V, by układ arbitrażu działał nienagannie, nawet jeśli cała "jego" szuflada straciła zasilanie główne.

### **3\. Arbitraż: Układ Rezerwacji Interfejsu (Speed-Independent)**

Zabezpiecza przed sytuacją kolizji na szynie. Żaden układ nie położy swoich sygnałów na interfejs, jeśli najpierw nie wystawi żądania **ZG (Zgłoszenie)** i nie otrzyma odpowiedzi **ZW (Zezwolenie)**.

* Pakiet interfejsu rozbudowany jest o *asynchronicznie sekwencyjny automat klasy Speed-Independent (SI)* zbudowany z bramek NAND (w tym z asynchronicznych przerzutników RS).  
* Układ składa się z modularnych bloków w kaskadzie. Ułożenie ich względem "górnego" terminatora TIF nadaje poszczególnym nadawcom w systemie ścisły, fizyczny **priorytet sprzętowy**.  
* Szybkość propagacji fali żądań i zezwoleń przez cały system zależy tu wyłącznie od praw fizyki w danym układzie (napięcie, temperatura, czas propagacji bramki logicznej w nanosekundach). Gdy napięcie rośnie, zjawisko arbitrażu dzieje się ułamek sekundy szybciej.

## **Podsumowanie**

Architektura CPU oraz magistrali Mery 400 to *sui generis* arcydzieło inżynierii opierające się o układy asynchroniczne. Jest dowodem na ogromny kunszt i elastyczność polskich projektantów w obliczu problemów ze skalowaniem i jakością dostępnych w tamtych czasach części. Dzięki takiemu projektowi, system pozwalał swobodnie dołączać bardzo wolne moduły na równi z wyjątkowo szybkimi, samoczynnie negocjując taktowanie i zachowując niezwykłą jak na lata 70\. stabilność.