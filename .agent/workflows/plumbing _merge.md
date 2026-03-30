---
description: Zautomatyzowany protokół niskopoziomowej synchronizacji gałęzi (Git Plumbing) dedykowany dla projektu OxTorch. Służy do wymuszania spójności między gałęziami dev i main
---

🛠️ OxTorch Plumbing Protocol (v1.0)Cel: Bezpieczna synchronizacja gałęzi Dev ↔ Main bez „Tech-Bro” błędów„Ad impossibilia nemo tenetur” (Nikt nie jest zobowiązany do rzeczy niemożliwych), ale Agent jest zobowiązany do przestrzegania ścieżek.1. Pre-flight Checks (Weryfikacja Topologii)Zanim wykonasz jakąkolwiek komendę git, musisz potwierdzić swoją lokalizację.ZASADA: Wszystkie operacje Git odbywają się w korzeniu projektu.Lokalizacja: pwd musi zwracać /my_data/gaussian_room/.Obecność Gita: ls -d .git musi istnieć w tej lokalizacji.Zakaz Podfolderów: Nigdy nie wywołuj komend synchronizacji będąc w vulkannn_rusted/ lub głębiej.2. Scenariusz A: „Zjebałem” (Praca wykonana na Main zamiast na Dev)Jeśli zmiany zostały zrobione na main, a powinny być na dev:# Będąc na dowolnej gałęzi, wymuś przesunięcie wskaźnika dev tam, gdzie jest main
git branch -f dev main

# Teraz obie gałęzie wskazują na ten sam (najnowszy) commit
# Możesz teraz bezpiecznie wrócić na dev
git checkout dev
3. Scenariusz B: Plumbing Sync (Wyrównanie Main do Dev)Gdy dev jest gotowy i chcesz, aby main stał się jego idealnym lustrem:# 1. Przejdź na main
git checkout main

# 2. Hard Reset (Plumbing method) - usuwa różnice i ustawia main na commit dev
git reset --hard dev

# 3. Wymuś aktualizację zdalną (jeśli potrzebne)
git push origin main --force
4. Metoda „Read-Tree” (Głęboki Plumbing)Używaj, gdy chcesz zaktualizować zawartość plików bez ruszania historii commitów na obecnym branchu (rzadkie, ale potężne):# Zastąp stan indexu i plików roboczych zawartością gałęzi dev
git read-tree -u -m dev
git commit -m "Plumbing: Sync with dev state via read-tree"
5. Algorytm Postępowania dla AgentaKażda interakcja z Gitem musi podążać tą ścieżką:Sprawdź Branch: git branch --show-current.Sprawdź Status: git status (czy nie ma plików, których zapomniałeś dodać?).Sync-First: Zawsze rób git pull origin <branch> przed rozpoczęciem pracy.No-Merge Policy: W tym projekcie preferujemy reset --hard lub branch -f, aby utrzymać czystą strukturę „Mera-Style” bez zbędnych commitów typu „Merge branch X into Y”.⚠️ Ostrzeżenie przed katastrofą„Errare humanum est, sed in errare perseverare diabolicum” (Błądzić jest rzeczą ludzką, ale trwać w błędzie jest rzeczą diabelską).Jeśli Agent zobaczy ostrzeżenie unable to rmdir lub directory not empty, oznacza to, że próbuje usunąć folder, który Git traktuje jako submoduł lub zagnieżdżone repo.STOP: Nie używaj rm -rf bez sprawdzenia, czy jesteś w głównym folderze /my_data/gaussian_room/.