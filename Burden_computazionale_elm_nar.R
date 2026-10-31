## =============================================================================
##
## Obiettivo: confrontare il tempo di addestramento di reti neurali feed-forward
## classiche (NNET, addestrate con BFGS) e di Extreme Learning Machines (ELM,
## addestrate con soluzione in forma chiusa) su serie storiche simulate dal
## modello di riferimento EXPAR(1) e EXPAR(2) (Exponential Autoregressive di 
## ordine 1 e 2), al variare della lunghezza della serie T e del numero di 
## neuroni nello strato nascosto m.
##
## =============================================================================

## -----------------------------------------------------------------------
## 0. Librerie necessarie
## -----------------------------------------------------------------------
# nnet    -> per addestrare le reti neurali feed-forward classiche (BFGS in C)
# ggplot2 -> per riprodurre un grafico stile Figura 1 del paper
# dplyr   -> per l'aggregazione dei risultati
if (!requireNamespace("nnet", quietly = TRUE)) install.packages("nnet")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(nnet)
library(ggplot2)
library(dplyr)

## -----------------------------------------------------------------------
## 1. Riproducibilita' dell'esperimento
## -----------------------------------------------------------------------
set.seed(2811)

## -----------------------------------------------------------------------
## 2. Simulazione del processo EXPAR(2) (eq. 20 del paper, Sezione 4)
## -----------------------------------------------------------------------
# Modello generatore dei dati usato dagli autori nell'esperimento sul costo
# computazionale (Sezione 4):
#
#   Y_t = a1(Y_{t-1}) * Y_{t-1} + a2(Y_{t-1}) * Y_{t-2} + e_t
#
#   con
#     a1(u) = 0.5 + 0.9 * exp(-u^2)
#     a2(u) = -0.8 + 1.8 * exp(-u^2)
#
# Gli errori sono generati come nel paper: e_t ~ T6 / sqrt(1.5), cioe' una
# t di Student con 6 gradi di liberta' riscalata per ottenere varianza
# unitaria (Var(T6) = 6/(6-2) = 1.5).

simula_expar2 <- function(T_len, burnin = 200) {
  # T_len  : lunghezza della serie storica desiderata (dopo il burn-in)
  # burnin : osservazioni iniziali scartate per eliminare l'effetto dei
  #          valori iniziali (transitorio del processo)
  
  n_tot <- T_len + burnin
  y <- numeric(n_tot)
  
  # errori i.i.d. da una t di Student con 6 g.l., riscalata per avere
  # varianza unitaria
  e <- rt(n_tot, df = 6) / sqrt(1.5)
  
  # valori iniziali arbitrari, piccoli, per partire vicino allo stato stazionario
  y[1] <- rnorm(1, sd = 0.1)
  y[2] <- rnorm(1, sd = 0.1)
  
  for (t in 3:n_tot) {
    y1 <- y[t - 1]
    y2 <- y[t - 2]
    
    a1 <- 0.5 + 0.9 * exp(-y1^2)   # coefficiente non lineare su Y_{t-1}
    a2 <- -0.8 + 1.8 * exp(-y1^2)  # coefficiente non lineare su Y_{t-2}
    
    y[t] <- a1 * y1 + a2 * y2 + e[t]
  }
  
  # si scarta il burn-in e si restituisce la serie di lunghezza T_len
  y[(burnin + 1):n_tot]
}

## -----------------------------------------------------------------------
## 3. Costruzione del dataset supervisionato (coppie input/target)
## -----------------------------------------------------------------------
# Con p = 2 fissato, il vettore di input per l'istante t e' x_{t-1} =
# (Y_{t-1}, Y_{t-2}) e il target e' Y_t (formula (4) del paper con p = 2).
# La stessa serie e lo stesso dataset vengono riutilizzati per tutte le
# configurazioni di m, esattamente come nel disegno sperimentale del paper.

crea_dataset_supervisionato <- function(y, p = 2) {
  T_len <- length(y)
  X <- matrix(NA_real_, nrow = T_len - p, ncol = p)
  for (j in 1:p) {
    X[, j] <- y[(p - j + 1):(T_len - j)]
  }
  target <- y[(p + 1):T_len]
  list(X = X, y = target)
}

## -----------------------------------------------------------------------
## 4. Funzioni di supporto: sigmoide e calcolo dell'RMSE
## -----------------------------------------------------------------------
sigmoide <- function(x) 1 / (1 + exp(-x))

rmse <- function(y_veri, y_predetti) sqrt(mean((y_veri - y_predetti)^2))

## -----------------------------------------------------------------------
## 5. Funzione di addestramento ELM (Extreme Learning Machine)
## -----------------------------------------------------------------------
# Implementa l'algoritmo della Sezione 3 del paper:
#  1) i pesi e i bias dello strato nascosto (W, b) sono generati casualmente
#     e NON vengono aggiornati (eq. 16);
#  2) i pesi di uscita c sono ottenuti in forma chiusa tramite una soluzione
#     di minimi quadrati regolarizzati (ridge, eq. 19):
#       c_hat = (H'H + I/eta)^(-1) H' y

addestra_elm <- function(X, y, m, eta = 1000) {
  # X   : matrice di input (T-p) x p
  # y   : vettore target (T-p)
  # m   : numero di neuroni nello strato nascosto
  # eta : parametro di regolarizzazione (fissato ad hoc, come discusso nel paper)
  
  p <- ncol(X)
  n <- nrow(X)
  
  # 1) generazione casuale dei pesi/bias dello strato nascosto
  W <- matrix(runif(m * p, -1, 1), nrow = p, ncol = m)   # pesi input->nascosto
  b <- runif(m, -1, 1)                                    # bias dei neuroni nascosti
  
  # 2) calcolo della matrice H dello strato nascosto (eq. 16)
  H <- sigmoide(X %*% W + matrix(b, n, m, byrow = TRUE))
  
  # 3) stima regolarizzata dei pesi di uscita c (eq. 19)
  I_m <- diag(m)
  c_hat <- solve(t(H) %*% H + I_m / eta) %*% t(H) %*% y
  
  list(W = W, b = b, c = c_hat, m = m)
}

# Funzione di predizione per un modello ELM gia' stimato: ricalcola la
# matrice nascosta H con gli stessi pesi casuali (W, b) usati in fase di
# stima e applica i pesi di uscita c.
predici_elm <- function(elm_fit, X) {
  n <- nrow(X)
  H <- sigmoide(X %*% elm_fit$W + matrix(elm_fit$b, n, elm_fit$m, byrow = TRUE))
  as.vector(H %*% elm_fit$c)
}

## -----------------------------------------------------------------------
## 6. Funzione di addestramento NNET con riavvii multipli e timing separato
## -----------------------------------------------------------------------
# Come nel paper, la rete viene riaddestrata n_restart volte con inizializ-
# zazioni casuali diverse (per evitare minimi locali) e si mantiene il
# modello con errore quadratico piu' basso sul training set.
#
# DUBBIO (1): per poter confrontare i tempi in due modi diversi, il tempo di
# CIASCUN riavvio viene cronometrato singolarmente con system.time(). La
# funzione restituisce quindi sia il tempo TOTALE (somma dei 10 riavvii,
# cioe' il costo "reale" per ottenere una rete affidabile, coerente con la
# Figura 1 del paper) sia il tempo del SOLO riavvio che ha prodotto il
# modello migliore (interpretabile come il costo di stima di "un singolo
# modello NNET", a parita' di condizioni con l'unico addestramento ELM).
#
# NOTA SU IN-SAMPLE VS OUT-OF-SAMPLE (rilevante per la Sezione 12):
# la selezione del modello migliore fra i 10 riavvii avviene SEMPRE in base
# all'errore di stima sul training set (modello_r$value, come nel paper).
# L'RMSE restituito dalla funzione, invece, puo' essere calcolato in due modi:
#   - se X_test/y_test NON sono forniti (default, usato in Sezione 8): RMSE
#     in-sample, cioe' sugli stessi dati di training (bonta' di adattamento);
#   - se X_test/y_test sono forniti (usato in Sezione 12): RMSE calcolato
#     sulle osservazioni di test, MAI viste durante la stima (capacita'
#     predittiva, out-of-sample).

addestra_nnet_con_restart <- function(X, y, m, n_restart = 10, X_test = NULL, y_test = NULL) {
  
  modello_migliore <- NULL
  errore_migliore <- Inf
  tempo_del_migliore <- NA_real_
  tempo_totale <- 0
  
  for (r in 1:n_restart) {
    
    tempo_riavvio <- system.time({
      modello_r <- nnet(
        x = X, y = y,
        size = m,          # numero di neuroni nello strato nascosto
        linout = TRUE,     # output lineare (problema di regressione)
        trace = FALSE,     # nessuna stampa a video durante l'ottimizzazione
        maxit = 500
      )
    })["elapsed"]
    
    tempo_totale <- tempo_totale + as.numeric(tempo_riavvio)
    
    errore_r <- modello_r$value   # valore finale della funzione obiettivo (SSE) sul training
    if (errore_r < errore_migliore) {
      errore_migliore <- errore_r
      modello_migliore <- modello_r
      tempo_del_migliore <- as.numeric(tempo_riavvio)
    }
  }
  
  # calcolo dell'RMSE: in-sample (default) oppure out-of-sample se e' stato
  # fornito un set di test (X_test, y_test)
  if (is.null(X_test) || is.null(y_test)) {
    rmse_migliore <- rmse(y, as.vector(fitted(modello_migliore)))
  } else {
    previsioni_test <- as.vector(predict(modello_migliore, X_test))
    rmse_migliore <- rmse(y_test, previsioni_test)
  }
  
  list(
    modello = modello_migliore,
    tempo_totale = tempo_totale,          # somma dei 10 riavvii (costo "pratico")
    tempo_migliore = tempo_del_migliore,  # tempo del solo riavvio vincente
    rmse = rmse_migliore
  )
}

## -----------------------------------------------------------------------
## 6.1 Funzione di ricerca del numero minimo di neuroni ELM per rientrare
##     entro una tolleranza relativa (default 5%) rispetto a un RMSE target
## -----------------------------------------------------------------------
# Usata sia in Sezione 8 (valutazione IN-SAMPLE: si passa lo stesso dataset
# come train e come test) sia in Sezione 12 (valutazione OUT-OF-SAMPLE: si
# passano due porzioni disgiunte della serie, ottenute con
# crea_dataset_train_test()).

trova_m_elm_per_target_accuratezza <- function(X_train, y_train, X_test, y_test, rmse_target,
                                               m_start = 4, m_step = 4,
                                               m_max = 300, tolleranza = 0.05) {
  # X_train, y_train : dati usati per stimare l'ELM
  # X_test, y_test   : dati su cui si valuta l'RMSE (possono coincidere con
  #                    X_train/y_train per una valutazione IN-SAMPLE, oppure
  #                    essere una porzione disgiunta per una valutazione
  #                    OUT-OF-SAMPLE)
  # rmse_target      : RMSE del modello NNET di riferimento
  # m_start          : numero di neuroni ELM da cui partire la ricerca (=4)
  # m_step           : incremento del numero di neuroni ad ogni iterazione
  # m_max            : limite massimo di neuroni, per evitare ricerche infinite
  # tolleranza       : scarto RELATIVO massimo accettato rispetto al target
  #                    (0.05 = soglia pari a rmse_target * 1.05, cioe' +5%)
  
  soglia <- rmse_target * (1 + tolleranza)
  m_corrente <- m_start
  
  repeat {
    elm_fit <- addestra_elm(X_train, y_train, m = m_corrente)
    rmse_elm_corrente <- rmse(y_test, predici_elm(elm_fit, X_test))
    
    raggiunto <- rmse_elm_corrente <= soglia
    if (raggiunto || m_corrente >= m_max) break
    
    m_corrente <- m_corrente + m_step
  }
  
  list(
    m_trovato = m_corrente,
    rmse_elm = rmse_elm_corrente,
    rmse_target = rmse_target,
    soglia = soglia,
    raggiunto = raggiunto
  )
}

## -----------------------------------------------------------------------
## 7. Disegno sperimentale
## -----------------------------------------------------------------------
T_vec  <- c(250, 500, 1000)     # lunghezze delle serie storiche
m_vec  <- c(4, 8, 12, 16, 20)   # dimensioni dello strato nascosto (i "5 modelli")
p_fisso <- 2                    # numero di INPUT fissato a 2, come richiesto
n_restart_nnet <- 10            # numero di riavvii della rete NNET

## Il paper genera complessivamente 5000 serie storiche, di lunghezza diversa
## (250, 500, 1000), e per CIASCUNA serie stima tutti e 5 i modelli (uno per
## ciascun valore di m). Qui si riproduce fedelmente questa impostazione,
## suddividendo le 5000 serie in modo equilibrato sui tre valori di T.
##
## NOTA: eseguire l'esperimento su 5000 serie x 5 valori di m x
## (10 riavvii NNET + 1 ELM) richiede un tempo di calcolo molto elevato.
## Per questo motivo e' incluso un interruttore "modalita_rapida": se TRUE,
## il numero di serie viene ridotto drasticamente per un test rapido dello
## script; per la replica fedele del paper impostare modalita_rapida <- FALSE.

modalita_rapida <- FALSE

n_serie_totali <- if (modalita_rapida) 300 else 5000
n_serie_per_T  <- round(n_serie_totali / length(T_vec))

cat(sprintf(
  "Modalita rapida: %s | Serie totali: %d | Serie per ciascun T: %d\n",
  modalita_rapida, n_serie_per_T * length(T_vec), n_serie_per_T
))

## -----------------------------------------------------------------------
## 8. Ciclo sperimentale: simulazione, addestramento, tempi e accuratezza
##    IN-SAMPLE (include la ricerca del numero minimo di neuroni ELM
##    necessari per rientrare nel 5% di RMSE rispetto alla NNET)
## -----------------------------------------------------------------------
# Per ciascuna lunghezza T si generano n_serie_per_T serie EXPAR(2). Per
# ogni serie, il dataset supervisionato viene costruito e riutilizzato per 
# stimare tutti e 5 i valori di m, sia con NNET sia con ELM: con grandezza dello
# strato nascosto variabile nell'insieme {4,8,12,16,20}".

risultati <- data.frame()

for (T_len in T_vec) {
  for (s in 1:n_serie_per_T) {
    
    # 8.1 simulazione di UNA serie EXPAR(2) di lunghezza T_len
    serie <- simula_expar2(T_len)
    
    # 8.2 costruzione del dataset supervisionato (riutilizzato per tutti gli m)
    dati <- crea_dataset_supervisionato(serie, p = p_fisso)
    
    for (m in m_vec) {
      
      # 8.3 addestramento NNET (10 riavvii, timing separato per ciascuno);
      # l'RMSE restituito e' IN-SAMPLE (X_test/y_test non forniti)
      esito_nnet <- addestra_nnet_con_restart(dati$X, dati$y, m = m, n_restart = n_restart_nnet)
      
      # 8.4 addestramento ELM ALLO STESSO m della NNET (confronto diretto,
      # utile per il confronto dei tempi) e misurazione del tempo
      tempo_elm <- system.time({
        elm_fit <- addestra_elm(dati$X, dati$y, m = m)
      })["elapsed"]
      rmse_elm <- rmse(dati$y, predici_elm(elm_fit, dati$X))
      
      # 8.5 ricerca IN-SAMPLE del numero minimo di neuroni ELM (> 20) che porta
      # l'RMSE entro il 5% (tolleranza relativa) di quello della NNET appena
      # stimata (m come dimensione di riferimento); si passa lo stesso dataset
      # come train e come test, ottenendo così una valutazione IN-SAMPLE
      esito_ricerca_is <- trova_m_elm_per_target_accuratezza(
        X_train = dati$X, y_train = dati$y,
        X_test  = dati$X, y_test  = dati$y,
        rmse_target = esito_nnet$rmse,
        m_start = 4, m_step = 4, m_max = 400, tolleranza = 0.05
      )
      
      # 8.6 salvataggio dei risultati della singola combinazione (T, serie, m)
      risultati <- rbind(
        risultati,
        data.frame(
          T = T_len,
          serie = s,
          m = m,
          p = p_fisso,
          tempo_nnet_totale     = esito_nnet$tempo_totale,
          tempo_nnet_migliore   = esito_nnet$tempo_migliore,
          tempo_elm             = as.numeric(tempo_elm),
          rmse_nnet             = esito_nnet$rmse,
          rmse_elm              = rmse_elm,               # ELM allo STESSO m della NNET
          m_elm_trovato_is      = esito_ricerca_is$m_trovato,     # ELM con m calibrato sull'obiettivo di accuratezza
          rmse_elm_ricerca_is   = esito_ricerca_is$rmse_elm,
          scarto_percentuale_is = 100 * (esito_ricerca_is$rmse_elm / esito_nnet$rmse - 1),
          soglia_raggiunta_is   = esito_ricerca_is$raggiunto
        )
      )
    }
  }
  cat(sprintf("Completato: T = %d (%d serie x %d valori di m)\n", T_len, n_serie_per_T, length(m_vec)))
}

## -----------------------------------------------------------------------
## 9. Calcolo dei rapporti tra i tempi di addestramento (NNET / ELM) e
##    riepilogo dell'accuratezza IN-SAMPLE (ex Sezione 13, ora qui integrata)
## -----------------------------------------------------------------------
# Per ciascuna cella (T, m) si calcolano due rapporti di tempo (vedi dubbio 1):
#  - rapporto_totale   = tempo medio dei 10 riavvii NNET / tempo medio ELM
#                        (coerente con la Figura 1 del paper)
#  - rapporto_migliore = tempo medio del singolo NNET migliore / tempo medio ELM
#                        (confronto "modello singolo contro modello singolo")
#
# Vengono inoltre aggregate le statistiche sulla ricerca IN-SAMPLE del numero
# minimo di neuroni ELM necessari (colonne *_is).
#
# Il punto 9.1, subito sotto, aggiunge il tempo di stima e il rapporto dei
# tempi dell'ELM stimato proprio al numero MEDIANO di neuroni individuato
# dalla ricerca IN-SAMPLE (m_mediano_necessario_is): questo calcolo richiede
# necessariamente l'aggregazione per cella gia' fatta qui in Sezione 9 (la
# mediana e' nota solo dopo aver raccolto tutte le n_serie_per_T repliche
# per quella cella), e non puo' quindi essere svolto direttamente dentro il
# ciclo per-replica della Sezione 8.

riepilogo_tempi <- risultati %>%
  group_by(T, m) %>%
  summarise(
    tempo_medio_nnet_totale     = mean(tempo_nnet_totale),
    tempo_medio_nnet_migliore   = mean(tempo_nnet_migliore),
    tempo_medio_elm             = mean(tempo_elm),
    rapporto_totale             = tempo_medio_nnet_totale / tempo_medio_elm,
    rapporto_migliore           = tempo_medio_nnet_migliore / tempo_medio_elm,
    rmse_medio_nnet             = mean(rmse_nnet),
    rmse_medio_elm              = mean(rmse_elm),
    m_medio_necessario_is       = mean(m_elm_trovato_is),
    m_mediano_necessario_is     = median(m_elm_trovato_is),
    quota_soglia_raggiunta_is   = mean(soglia_raggiunta_is),
    scarto_medio_percentuale_is = mean(scarto_percentuale_is),
    .groups = "drop"
  )

print(riepilogo_tempi)

## -----------------------------------------------------------------------
## 9.1 Tempo di stima dell'ELM al numero MEDIANO di neuroni necessari
##     IN-SAMPLE, e rapporto con il tempo del singolo NNET migliore
## -----------------------------------------------------------------------
# Analogo, per l'accuratezza IN-SAMPLE, di quanto gia' fatto al punto 12.3
# per l'accuratezza OUT-OF-SAMPLE: per ciascuna cella (T, m) si stima
# nuovamente un ELM, questa volta fissando il numero di neuroni pari al
# valore MEDIANO individuato dalla ricerca IN-SAMPLE della Sezione 8
# (colonna m_mediano_necessario_is di riepilogo_tempi, arrotondato
# all'intero piu' vicino). Si misura il relativo tempo di stima e lo si
# confronta con il tempo medio del singolo NNET migliore ottenuto nella
# stessa cella, calcolando:
#
#   rapporto_tempo_mediano_is = tempo_medio_nnet_migliore
#                                 / tempo medio ELM (con m = m mediano
#                                   necessario IN-SAMPLE)
#
# In questo modo la tabella dei risultati IN-SAMPLE (riepilogo_tempi)
# riporta, oltre al numero mediano di neuroni necessari, anche il relativo
# costo computazionale dell'ELM a quella dimensione e il confronto con la
# NNET di riferimento.
#

if (!exists("riepilogo_tempi")) {
  percorso_salvataggio <- "riepilogo_tempi_sezione9.rds"
  if (file.exists(percorso_salvataggio)) {
    riepilogo_tempi <- readRDS(percorso_salvataggio)
    cat(sprintf(
      "riepilogo_tempi ricaricato da '%s' (Sezioni 7-9 non rieseguite in questa sessione).\n",
      percorso_salvataggio
    ))
  } else {
    stop(paste(
      "L'oggetto 'riepilogo_tempi' non e' presente in memoria e non e' stato",
      "trovato nessun file salvato ('riepilogo_tempi_sezione9.rds').",
      "Eseguire prima le Sezioni 7-9 (che lo calcolano e, se abilitato,",
      "lo salvano in Sezione 11), oppure fornire un file di risultati",
      "salvato da un'esecuzione precedente."
    ))
  }
}

n_rep_tempo_elm_mediano_is <- n_serie_per_T  # stesso numero di repliche usato in Sezione 8


tempo_elm_al_mediano_is <- data.frame()

for (i in seq_len(nrow(riepilogo_tempi))) {
  
  T_len     <- riepilogo_tempi$T[i]
  m_val     <- riepilogo_tempi$m[i]
  m_mediano <- round(riepilogo_tempi$m_mediano_necessario_is[i])  # arrotondamento all'intero piu' vicino
  
  tempi_replica <- numeric(n_rep_tempo_elm_mediano_is)
  
  for (rep in 1:n_rep_tempo_elm_mediano_is) {
    serie <- simula_expar2(T_len)
    dati <- crea_dataset_supervisionato(serie, p = p_fisso)
    
    tempo_rep <- system.time({
      addestra_elm(dati$X, dati$y, m = m_mediano)
    })["elapsed"]
    
    tempi_replica[rep] <- as.numeric(tempo_rep)
  }
  
  tempo_elm_al_mediano_is <- rbind(
    tempo_elm_al_mediano_is,
    data.frame(
      T = T_len,
      m = m_val,
      m_mediano_usato_is = m_mediano,
      tempo_medio_elm_al_mediano_is = mean(tempi_replica)
    )
  )
}

# si unisce il tempo ELM (calcolato con m pari al numero mediano necessario
# IN-SAMPLE) al riepilogo dei tempi, e si calcola il rapporto richiesto
riepilogo_tempi <- riepilogo_tempi %>%
  left_join(tempo_elm_al_mediano_is, by = c("T", "m")) %>%
  mutate(
    rapporto_tempo_mediano_is = tempo_medio_nnet_migliore / tempo_medio_elm_al_mediano_is
  )

print(riepilogo_tempi)

# Interpretazione: rapporto_tempo_mediano_is confronta il tempo del singolo
# NNET migliore con il tempo di stima di un ELM avente proprio il numero
# mediano di neuroni necessario, IN-SAMPLE, per rientrare entro il 5% del
# suo RMSE. Un valore superiore a 1 indica che, anche dovendo usare piu'
# neuroni per compensare la mancata ottimizzazione dei pesi nascosti, l'ELM
# resta piu' veloce da stimare rispetto al singolo modello NNET migliore.

## 10. Grafici riassuntivi
## -----------------------------------------------------------------------
# Vengono prodotti due grafici, uno per ciascuna delle due definizioni di
# rapporto discusse al punto 9, cosi' da rendere esplicito l'effetto della
# scelta metodologica sul risultato finale.

dati_grafico <- riepilogo_tempi %>%
  select(T, m, rapporto_totale, rapporto_migliore) %>%
  tidyr::pivot_longer(
    cols = c(rapporto_totale, rapporto_migliore),
    names_to = "tipo_rapporto",
    values_to = "rapporto"
  ) %>%
  mutate(
    tipo_rapporto = recode(
      tipo_rapporto,
      rapporto_totale   = "NNET: 10 riavvii (paper)",
      rapporto_migliore = "NNET: solo il modello migliore"
    )
  )

grafico <- ggplot(dati_grafico, aes(x = factor(T), y = rapporto)) +
  geom_col(position = position_dodge()) +
  facet_wrap(~ m, nrow = 1, labeller = label_both) +
  labs(
    title = "Rapporto tra i tempi di addestramento NNET ed ELM - EXPAR(2)",
    x = "Lunghezza della serie storica (T)",
    y = "Rapporto tempo di calcolo NNET / ELM",
    fill = "Definizione del tempo NNET"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(grafico)

## =============================================================================
## 11. ACCURATEZZA A PARITA' DI PERFORMANCE: quanti neuroni servono a un ELM
##     per avvicinarsi (entro il 5%) all'RMSE di una NNET stimata con BFGS?
## =============================================================================
# DUBBIO (3): a differenza dell'esperimento sui tempi (dove m e' fissato e si
# confrontano i tempi), qui si fissa un obiettivo di ACCURATEZZA e si cerca,
# per un dato modello NNET di riferimento, il piu' piccolo numero di neuroni
# nascosti m (> 20, come richiesto) per cui un ELM raggiunge un RMSE non
# superiore al 5% in piu' rispetto a quello della rete NNET.
#
# IMPORTANTE - RMSE IN-SAMPLE vs OUT-OF-SAMPLE: nella Sezione 8 l'RMSE viene
# calcolato IN-SAMPLE, cioe' sugli stessi dati usati per la stima (bonta' di
# adattamento), ed e' rimasto invariato. In QUESTA sezione, invece, l'RMSE
# viene calcolato OUT-OF-SAMPLE: per ciascuna serie simulata si riservano le
# ultime h_test_oos osservazioni come insieme di TEST, mai utilizzate in fase
# di stima (ne' per la NNET, ne' per l'ELM). In questo modo l'RMSE misura la
# capacita' predittiva dei modelli su dati non visti, e non la loro semplice
# capacita' di adattarsi ai dati di stima. Questo e' rilevante soprattutto
# qui, perche' la ricerca del numero minimo di neuroni ELM aumenta m fino a
# valori molto grandi (fino a m_max): con un RMSE in-sample, aumentare m
# ridurrebbe l'errore quasi meccanicamente (sovradattamento), rendendo il
# confronto poco informativo sulla reale capacita' predittiva.
#
# Procedura, per una singola serie storica:
#   1) si divide la serie in una porzione di TRAINING e una di TEST (le
#      ultime h_test_oos osservazioni, mai usate in stima);
#   2) si stima il modello NNET di riferimento sul solo training (10
#      riavvii, si tiene il migliore) e se ne calcola l'RMSE sul TEST:
#      rmse_nnet (fuori campione);
#   3) si definisce la soglia di accettabilita': soglia = rmse_nnet * 1.05;
#   4) si stima un ELM sul solo training, partendo da m = m_start (=4); se
#      il suo RMSE calcolato sul TEST non e' entro la soglia, si aumenta m
#      di uno step (m_step) e si riprova, fino al raggiungimento della
#      soglia o a un numero massimo m_max di neuroni (limite di sicurezza).

## -----------------------------------------------------------------------
## 11.0 Split treno/test temporale (solo per questa sezione)
## -----------------------------------------------------------------------
# Dato che la serie e' autoregressiva, lo split rispetta l'ordine temporale:
# le ultime h_test_oos coppie (x_{t-1}, y_t) del dataset supervisionato
# vengono riservate al test, tutte le precedenti vengono usate per il
# training. In questo modo il test valuta previsioni "1-step ahead" fatte
# con valori ritardati realmente osservati (mai stimati ricorsivamente).

crea_dataset_train_test <- function(y, p = 2, h_test = 20) {
  # y      : serie storica simulata
  # p      : numero di ritardi (fissato a 2 in questo script)
  # h_test : numero di osservazioni finali riservate al test
  
  ds <- crea_dataset_supervisionato(y, p)
  n_tot <- nrow(ds$X)
  
  if (h_test >= n_tot) {
    stop("h_test troppo grande rispetto alla lunghezza della serie disponibile")
  }
  
  idx_train <- 1:(n_tot - h_test)
  idx_test  <- (n_tot - h_test + 1):n_tot
  
  list(
    X_train = ds$X[idx_train, , drop = FALSE],
    y_train = ds$y[idx_train],
    X_test  = ds$X[idx_test, , drop = FALSE],
    y_test  = ds$y[idx_test]
  )
}

## NOTA: la funzione trova_m_elm_per_target_accuratezza(), usata sia qui sia
## nella Sezione 8, e' definita subito dopo la Sezione 6, perche' serve gia'
## nel ciclo della Sezione 8 (ricerca IN-SAMPLE del numero minimo di neuroni
## ELM necessari) prima ancora di arrivare a questa sezione.

## -----------------------------------------------------------------------
## 11.1 Esempio di applicazione: la dimensione NNET di riferimento varia
##      lungo l'intera griglia m_vec = {4, 8, 12, 16, 20}
## -----------------------------------------------------------------------
# Rispetto alla versione precedente, la rete NNET di riferimento non ha piu'
# un'unica dimensione fissa (m = 20), ma varia lungo tutta la griglia usata
# nell'esperimento sui tempi (Sezioni 7-10): m_riferimento in {4,8,12,16,20}.
# Per ciascuna combinazione (T, m_riferimento) si ripete piu' volte la
# ricerca del piu' piccolo numero di neuroni ELM (> 20) che permette di
# rientrare entro il 5% dell'RMSE della corrispondente NNET di riferimento.
# Si conserva inoltre, per ogni replica, il tempo del singolo NNET migliore
# (gia' calcolato nella Sezione 6), utile al confronto dei tempi al punto 12.3.

n_rep_accuratezza <- if (modalita_rapida) 5 else 5000   # ridurre/aumentare secondo necessita'
h_test_oos <- 20   # numero di osservazioni finali di ciascuna serie riservate al test
# (mai usate in stima ne' per NNET ne' per ELM)

risultati_accuratezza <- data.frame()

for (T_len in T_vec) {
  for (m_rif in m_vec) {
    for (rep in 1:n_rep_accuratezza) {
      
      serie <- simula_expar2(T_len)
      
      # split temporale treno/test: il test e' composto dalle ultime
      # h_test_oos osservazioni della serie, mai viste in fase di stima
      dati <- crea_dataset_train_test(serie, p = p_fisso, h_test = h_test_oos)
      
      # modello NNET di riferimento, stimato SOLO sul training, con
      # dimensione dello strato nascosto pari al valore corrente di m_vec;
      # l'RMSE restituito e' calcolato sul TEST (out-of-sample), passando
      # X_test/y_test alla funzione
      esito_nnet <- addestra_nnet_con_restart(
        dati$X_train, dati$y_train, m = m_rif, n_restart = n_restart_nnet,
        X_test = dati$X_test, y_test = dati$y_test
      )
      
      # ricerca del numero minimo di neuroni ELM per rientrare nel 5% di
      # RMSE out-of-sample: l'ELM viene stimato sul training e valutato sul
      # test, esattamente come la NNET di riferimento
      esito_ricerca <- trova_m_elm_per_target_accuratezza(
        X_train = dati$X_train, y_train = dati$y_train,
        X_test = dati$X_test, y_test = dati$y_test,
        rmse_target = esito_nnet$rmse,
        m_start = 4, m_step = 4, m_max = 400, tolleranza = 0.05
      )
      
      risultati_accuratezza <- rbind(
        risultati_accuratezza,
        data.frame(
          T = T_len,
          m_riferimento = m_rif,
          replica = rep,
          rmse_nnet_oos = esito_nnet$rmse,                   # RMSE NNET out-of-sample (sul test)
          tempo_nnet_migliore = esito_nnet$tempo_migliore,    # tempo del solo NNET vincente (Sezione 6)
          m_elm_trovato = esito_ricerca$m_trovato,
          rmse_elm_oos = esito_ricerca$rmse_elm,              # RMSE ELM out-of-sample (sul test)
          scarto_percentuale = 100 * (esito_ricerca$rmse_elm / esito_nnet$rmse - 1),
          soglia_raggiunta = esito_ricerca$raggiunto
        )
      )
    }
  }
  cat(sprintf("Ricerca accuratezza completata per T = %d (tutte le dimensioni m di riferimento)\n", T_len))
}

## -----------------------------------------------------------------------
## 11.2 Riepilogo: numero di neuroni ELM necessari, per T e per m di riferimento
## -----------------------------------------------------------------------
# I valori di RMSE utilizzati per costruire m_elm_trovato (e quindi anche
# scarto_medio_percentuale e quota_soglia_raggiunta) sono OUT-OF-SAMPLE, cioe'
# calcolati sulle ultime h_test_oos osservazioni di ciascuna serie, mai usate
# in fase di stima.
riepilogo_accuratezza <- risultati_accuratezza %>%
  group_by(T, m_riferimento) %>%
  summarise(
    m_medio_necessario        = mean(m_elm_trovato),
    m_mediano_necessario      = median(m_elm_trovato),
    quota_soglia_raggiunta    = mean(soglia_raggiunta),   # frazione di casi risolti entro m_max
    scarto_medio_percentuale  = mean(scarto_percentuale),
    tempo_medio_nnet_migliore = mean(tempo_nnet_migliore),
    rmse_mediano_nnet           = median(rmse_nnet_oos),
    rmse_mediano_elm            = median(rmse_elm_oos),
    .groups = "drop"
  )

print(riepilogo_accuratezza)

## -----------------------------------------------------------------------
## 11.3 Tempo di stima dell'ELM in corrispondenza del numero MEDIANO di
##      neuroni necessari, e rapporto con il tempo del singolo NNET migliore
## -----------------------------------------------------------------------
# Per ciascuna cella (T, m_riferimento) si stima nuovamente un ELM, questa
# volta fissando il numero di neuroni pari al valore MEDIANO individuato al
# punto 12.2 (arrotondato all'intero piu' vicino): m_mediano_necessario. Si
# misura il relativo tempo di stima e lo si confronta con il tempo medio del
# singolo NNET migliore ottenuto nella stessa cella, calcolando:
#
#   rapporto_tempo_mediano = tempo medio NNET migliore
#                             / tempo medio ELM (con m = m mediano necessario)
#
# Questo risponde direttamente alla richiesta di mantenere, nel riepilogo
# finale, sia il risultato di accuratezza (m mediano necessario) sia il
# corrispondente confronto dei tempi di stima.

n_rep_tempo_elm_mediano <- n_rep_accuratezza  # stesso numero di repliche usato al punto 12.1

tempo_elm_al_mediano <- data.frame()

for (i in seq_len(nrow(riepilogo_accuratezza))) {
  
  T_len     <- riepilogo_accuratezza$T[i]
  m_rif     <- riepilogo_accuratezza$m_riferimento[i]
  m_mediano <- round(riepilogo_accuratezza$m_mediano_necessario[i])  # arrotondamento all'intero piu' vicino
  
  tempi_replica <- numeric(n_rep_tempo_elm_mediano)
  
  for (rep in 1:n_rep_tempo_elm_mediano) {
    serie <- simula_expar2(T_len)
    dati <- crea_dataset_supervisionato(serie, p = p_fisso)
    
    tempo_rep <- system.time({
      addestra_elm(dati$X, dati$y, m = m_mediano)
    })["elapsed"]
    
    tempi_replica[rep] <- as.numeric(tempo_rep)
  }
  
  tempo_elm_al_mediano <- rbind(
    tempo_elm_al_mediano,
    data.frame(
      T = T_len,
      m_riferimento = m_rif,
      m_mediano_usato = m_mediano,
      tempo_medio_elm_al_mediano = mean(tempi_replica)
    )
  )
}

# si unisce il tempo ELM (calcolato con m pari al numero mediano necessario)
# al riepilogo di accuratezza, e si calcola il rapporto richiesto
riepilogo_accuratezza <- riepilogo_accuratezza %>%
  left_join(tempo_elm_al_mediano, by = c("T", "m_riferimento")) %>%
  mutate(
    rapporto_tempo_mediano = tempo_medio_nnet_migliore / tempo_medio_elm_al_mediano
  )

print(riepilogo_accuratezza)

# Interpretazione: per ciascuna combinazione di lunghezza della serie T e
# dimensione NNET di riferimento m_riferimento, la tabella mostra:
#  - m_mediano_necessario: quanti neuroni nascosti servono, in mediana, a un
#    ELM per raggiungere un'accuratezza (RMSE) entro il 5% di quella della
#    NNET di riferimento;
#  - rapporto_tempo_mediano: il rapporto tra il tempo del singolo NNET
#    migliore e il tempo di stima di un ELM avente proprio m_mediano_necessario
#    neuroni. Un valore ancora superiore a 1 indica che, anche dovendo usare
#    piu' neuroni per compensare la mancata ottimizzazione dei pesi, l'ELM
#    resta piu' veloce da stimare rispetto al singolo modello NNET migliore.


## -----------------------------------------------------------------------
## 11.4 Confronto diretto: numero di neuroni ELM necessari, in-sample
##      (Sezione 8/9, colonna m_mediano_necessario_is di riepilogo_tempi)
##      vs out-of-sample (questa sezione, m_mediano_necessario)
## -----------------------------------------------------------------------
# NOTA: la ricerca IN-SAMPLE del numero minimo di neuroni ELM, che in una
# versione precedente dello script costituiva una sezione a parte (Sezione
# 13, poi rimossa), e' ora svolta direttamente nella Sezione 8, nello stesso
# ciclo che stima anche i tempi di addestramento, perche' utilizza
# esattamente le stesse funzioni e la stessa valutazione IN-SAMPLE gia'
# presenti in quella sezione (nessun dato viene mai riservato al test). I
# relativi risultati aggregati si trovano quindi in riepilogo_tempi (Sezione
# 9), colonna m_mediano_necessario_is. Qui li si confronta direttamente con
# l'analogo risultato OUT-OF-SAMPLE appena calcolato in questa sezione.

confronto_in_out_sample <- riepilogo_accuratezza %>%
  select(T, m_riferimento, m_mediano_necessario_oos = m_mediano_necessario) %>%
  left_join(
    riepilogo_tempi %>%
      select(T, m, m_mediano_necessario_is),
    by = c("T" = "T", "m_riferimento" = "m")
  ) %>%
  mutate(
    differenza_m = m_mediano_necessario_oos - m_mediano_necessario_is
  )

print(confronto_in_out_sample)

# Interpretazione: se differenza_m e' sistematicamente negativa (cioe' serve
# un m maggiore IN-SAMPLE che OUT-OF-SAMPLE), significa che la ricerca
# in-sample e' resa piu' severa dal sovradattamento della NNET di
# riferimento: il suo RMSE di training e' artificiosamente basso, il che
# alza la soglia richiesta all'ELM e lo costringe ad usare piu' neuroni per
# raggiungerla, pur non riflettendo una reale differenza di capacita'
# predittiva. Una differenza_m positiva indicherebbe invece il comportamento
# opposto (soglia out-of-sample piu' severa).




## =============================================================================
## 12. RIPETIZIONE DELL'INTERA ANALISI PER UN MODELLO EXPAR(1)
## =============================================================================
# Obiettivo: ripercorrere l'intera analisi svolta finora (costo computazionale
# NNET vs ELM e accuratezza IN-SAMPLE, Sezioni 7-9; accuratezza OUT-OF-SAMPLE,
# Sezione 12) utilizzando come modello generatore dei dati un EXPAR(1) invece
# dell'EXPAR(2), MA con gli stessi parametri gia' usati per l'EXPAR(2): si
# riutilizza lo stesso coefficiente non lineare
#
#   a1(u) = 0.5 + 0.9 * exp(-u^2)
#
# e la stessa distribuzione dell'errore (T di Student con 6 g.l., riscalata),
# semplicemente eliminando il secondo ritardo e il relativo coefficiente
# a2(.), che nell'EXPAR(1) non esiste:
#
#   Y_t = a1(Y_{t-1}) * Y_{t-1} + e_t
#
# NUMERO DI NEURONI DI INPUT (p): a differenza del resto dello script, dove
# p e' fissato a 2 (variabile p_fisso, Sezione 7) indipendentemente dal vero
# ordine del processo generatore, in questa sezione il numero di neuroni di
# INPUT usato per addestrare NNET ed ELM viene fatto coincidere con l'ordine
# del modello teorico di riferimento: essendo l'EXPAR(1) un processo di
# ordine 1, si utilizza p_expar1 = 1 (un solo neurone di input, x_{t-1} =
# Y_{t-1}).
#
# NOTA SUI PARAMETRI DI RICERCA ELM: la ricerca IN-SAMPLE (punto 12.2, come
# in Sezione 8) mantiene gli stessi valori di partenza usati altrove nello
# script (m_start = 4, m_step = 4).

## -----------------------------------------------------------------------
## 12.0 Simulazione del processo EXPAR(1), con gli stessi parametri
##      dell'EXPAR(2) gia' definito (Sezione 2)
## -----------------------------------------------------------------------
simula_expar1 <- function(T_len, burnin = 200) {
  # T_len  : lunghezza della serie storica desiderata (dopo il burn-in)
  # burnin : osservazioni iniziali scartate per eliminare l'effetto dei
  #          valori iniziali (transitorio del processo)
  
  n_tot <- T_len + burnin
  y <- numeric(n_tot)
  
  # stessa distribuzione degli errori usata per l'EXPAR(2)
  e <- rt(n_tot, df = 6) / sqrt(1.5)
  
  y[1] <- rnorm(1, sd = 0.1)
  
  for (t in 2:n_tot) {
    y1 <- y[t - 1]
    
    # stesso coefficiente a1(.) usato nell'EXPAR(2) (Sezione 2); qui non
    # compare alcun termine a2(.) * Y_{t-2}, perche' il processo e' di ordine 1
    a1 <- 0.5 + 0.9 * exp(-y1^2)
    
    y[t] <- a1 * y1 + e[t]
  }
  
  y[(burnin + 1):n_tot]
}

## -----------------------------------------------------------------------
## 12.1 Disegno sperimentale per l'EXPAR(1)
## -----------------------------------------------------------------------
# Si riutilizzano T_vec, m_vec, n_restart_nnet, n_serie_per_T, n_rep_accuratezza
# e h_test_oos gia' definiti in precedenza (Sezioni 7 e 12); cambia solo il
# numero di ritardi in input, qui pari all'ordine del vero modello EXPAR(1).

p_expar1 <- 1   # un solo neurone di input, pari all'ordine del modello teorico EXPAR(1)

## -----------------------------------------------------------------------
## 12.2 Ciclo sperimentale: tempi di addestramento + accuratezza IN-SAMPLE
##      per l'EXPAR(1) (equivalente Sezioni 8-9)
## -----------------------------------------------------------------------
risultati_expar1 <- data.frame()

for (T_len in T_vec) {
  for (s in 1:n_serie_per_T) {
    
    serie <- simula_expar1(T_len)
    dati <- crea_dataset_supervisionato(serie, p = p_expar1)
    
    for (m in m_vec) {
      
      esito_nnet <- addestra_nnet_con_restart(dati$X, dati$y, m = m, n_restart = n_restart_nnet)
      
      tempo_elm <- system.time({
        elm_fit <- addestra_elm(dati$X, dati$y, m = m)
      })["elapsed"]
      rmse_elm <- rmse(dati$y, predici_elm(elm_fit, dati$X))
      
      # ricerca IN-SAMPLE del numero minimo di neuroni ELM (stessi parametri
      # di partenza usati in Sezione 8: m_start = 4, m_step = 4)
      esito_ricerca_is <- trova_m_elm_per_target_accuratezza(
        X_train = dati$X, y_train = dati$y,
        X_test  = dati$X, y_test  = dati$y,
        rmse_target = esito_nnet$rmse,
        m_start = 4, m_step = 4, m_max = 400, tolleranza = 0.05
      )
      
      risultati_expar1 <- rbind(
        risultati_expar1,
        data.frame(
          T = T_len,
          serie = s,
          m = m,
          p = p_expar1,
          tempo_nnet_totale     = esito_nnet$tempo_totale,
          tempo_nnet_migliore   = esito_nnet$tempo_migliore,
          tempo_elm             = as.numeric(tempo_elm),
          rmse_nnet             = esito_nnet$rmse,
          rmse_elm              = rmse_elm,
          m_elm_trovato_is      = esito_ricerca_is$m_trovato,
          rmse_elm_ricerca_is   = esito_ricerca_is$rmse_elm,
          scarto_percentuale_is = 100 * (esito_ricerca_is$rmse_elm / esito_nnet$rmse - 1),
          soglia_raggiunta_is   = esito_ricerca_is$raggiunto
        )
      )
    }
  }
  cat(sprintf("[EXPAR(1)] Tempi + accuratezza IN-SAMPLE completati per T = %d\n", T_len))
}

riepilogo_tempi_expar1 <- risultati_expar1 %>%
  group_by(T, m) %>%
  summarise(
    tempo_medio_nnet_totale     = mean(tempo_nnet_totale),
    tempo_medio_nnet_migliore   = mean(tempo_nnet_migliore),
    tempo_medio_elm             = mean(tempo_elm),
    rapporto_totale             = tempo_medio_nnet_totale / tempo_medio_elm,
    rapporto_migliore           = tempo_medio_nnet_migliore / tempo_medio_elm,
    rmse_medio_nnet             = mean(rmse_nnet),
    rmse_medio_elm              = mean(rmse_elm),
    m_medio_necessario_is       = mean(m_elm_trovato_is),
    m_mediano_necessario_is     = median(m_elm_trovato_is),
    quota_soglia_raggiunta_is   = mean(soglia_raggiunta_is),
    scarto_medio_percentuale_is = mean(scarto_percentuale_is),
    .groups = "drop"
  )

print(riepilogo_tempi_expar1)

## -----------------------------------------------------------------------
## 12.3 Grafico riassuntivo dei tempi per l'EXPAR(1) (equivalente Sezione 10)
## -----------------------------------------------------------------------
dati_grafico_expar1 <- riepilogo_tempi_expar1 %>%
  select(T, m, rapporto_migliore) %>%
  tidyr::pivot_longer(
    cols = rapporto_migliore,
    names_to = "tipo_rapporto",
    values_to = "rapporto"
  ) %>%
  mutate(
    tipo_rapporto = recode(
      tipo_rapporto,
      rapporto_totale   = "NNET: 10 riavvii (paper)",
      rapporto_migliore = "NNET: solo il modello migliore"
    )
  )

grafico_expar1 <- ggplot(dati_grafico_expar1, aes(x = factor(T), y = rapporto)) +
  geom_col(position = position_dodge()) +
  facet_wrap(~ m, nrow = 1, labeller = label_both) +
  labs(
    title = "Rapporto tra i tempi di addestramento NNET ed ELM - EXPAR(1)",
    x = "Lunghezza della serie storica (T)",
    y = "Rapporto tempo di calcolo NNET / ELM",
    fill = "Definizione del tempo NNET"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(grafico_expar1)

## -----------------------------------------------------------------------
## 12.4 Accuratezza OUT-OF-SAMPLE per l'EXPAR(1) (equivalente Sezione 12)
## -----------------------------------------------------------------------

risultati_accuratezza_oos_expar1 <- data.frame()

for (T_len in T_vec) {
  for (m_rif in m_vec) {
    for (rep in 1:n_rep_accuratezza) {
      
      serie <- simula_expar1(T_len)
      
      # split temporale treno/test: le ultime h_test_oos osservazioni sono
      # riservate al test, mai viste in fase di stima
      dati <- crea_dataset_train_test(serie, p = p_expar1, h_test = h_test_oos)
      
      esito_nnet <- addestra_nnet_con_restart(
        dati$X_train, dati$y_train, m = m_rif, n_restart = n_restart_nnet,
        X_test = dati$X_test, y_test = dati$y_test
      )
      
      esito_ricerca <- trova_m_elm_per_target_accuratezza(
        X_train = dati$X_train, y_train = dati$y_train,
        X_test = dati$X_test, y_test = dati$y_test,
        rmse_target = esito_nnet$rmse,
        m_start = 4, m_step = 4, m_max = 400, tolleranza = 0.05
      )
      
      risultati_accuratezza_oos_expar1 <- rbind(
        risultati_accuratezza_oos_expar1,
        data.frame(
          T = T_len,
          m_riferimento = m_rif,
          replica = rep,
          rmse_nnet_oos = esito_nnet$rmse,
          tempo_nnet_migliore = esito_nnet$tempo_migliore,
          m_elm_trovato = esito_ricerca$m_trovato,
          rmse_elm_oos = esito_ricerca$rmse_elm,
          scarto_percentuale = 100 * (esito_ricerca$rmse_elm / esito_nnet$rmse - 1),
          soglia_raggiunta = esito_ricerca$raggiunto
        )
      )
    }
  }
  cat(sprintf("[EXPAR(1)] Accuratezza OUT-OF-SAMPLE completata per T = %d\n", T_len))
}

riepilogo_accuratezza_oos_expar1 <- risultati_accuratezza_oos_expar1 %>%
  group_by(T, m_riferimento) %>%
  summarise(
    m_medio_necessario        = mean(m_elm_trovato),
    m_mediano_necessario      = median(m_elm_trovato),
    quota_soglia_raggiunta    = mean(soglia_raggiunta),
    scarto_medio_percentuale  = mean(scarto_percentuale),
    tempo_medio_nnet_migliore = mean(tempo_nnet_migliore),
    rmse_mediano_nnet           = median(rmse_nnet_oos),
    rmse_mediano_elm            = median(rmse_elm_oos),
    .groups = "drop"
  )

print(riepilogo_accuratezza_oos_expar1)
library(dplyr)
library(tidyr)
library(ggplot2)

riepilogo_rmse_lungo_expar1 <- riepilogo_accuratezza_oos_expar1 %>%
  select(T, m = m_riferimento, rmse_mediano_nnet, rmse_mediano_elm) %>%
  pivot_longer(
    cols      = c(rmse_mediano_nnet, rmse_mediano_elm),
    names_to  = "modello",
    values_to = "rmse_mediano"
  ) %>%
  mutate(
    modello = recode(modello,
                     rmse_mediano_nnet = "NNET",
                     rmse_mediano_elm  = "ELM")
  )

print(riepilogo_rmse_lungo_expar1)

# --- Grafico a barre: RMSE mediano per T e per m, confronto NNET vs ELM ---
grafico_rmse_oos_expar1 <- ggplot(
  riepilogo_rmse_lungo_expar1,
  aes(x = factor(T), y = rmse_mediano, fill = modello)
) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ m, nrow = 1, labeller = label_both) +
  labs(
    title = "RMSE medio out-of-sample: NNET vs ELM - EXPAR(1)",
    x = "Lunghezza della serie storica (T)",
    y = "RMSE medio",
    fill = "Modello"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(grafico_rmse_oos_expar1)
## -----------------------------------------------------------------------
## 12.5 Confronto diretto: numero di neuroni ELM necessari, in-sample vs
##      out-of-sample, per l'EXPAR(1) (equivalente Sezione 12.4)
## -----------------------------------------------------------------------
confronto_in_out_sample_expar1 <- riepilogo_accuratezza_oos_expar1 %>%
  select(T, m_riferimento, m_mediano_necessario_oos = m_mediano_necessario) %>%
  left_join(
    riepilogo_tempi_expar1 %>% select(T, m, m_mediano_necessario_is),
    by = c("T" = "T", "m_riferimento" = "m")
  ) %>%
  mutate(differenza_m = m_mediano_necessario_oos - m_mediano_necessario_is)

print(confronto_in_out_sample_expar1)

## -----------------------------------------------------------------------
## 12.6 Confronto diretto EXPAR(1) vs EXPAR(2), sullo stesso indicatore
##      (rapporto dei tempi "migliore", per T e m)
## -----------------------------------------------------------------------
# Si confronta il rapporto_migliore (tempo del singolo NNET migliore diviso
# il tempo dell'ELM) ottenuto per l'EXPAR(1) (p = 1) con quello gia'
# calcolato per l'EXPAR(2) (p = p_fisso = 2) nella Sezione 9 (oggetto
# riepilogo_tempi), per verificare se il vantaggio computazionale dell'ELM
# sia sensibile alla complessita' del vero processo generatore dei dati.
# Il confronto e' fatto a parita' di (T, m): il numero di neuroni di INPUT p
# resta diverso tra i due modelli (1 per l'EXPAR(1), 2 per l'EXPAR(2)), come
# richiesto, dato che ciascuno usa il proprio ordine teorico di riferimento.

confronto_expar1_expar2 <- riepilogo_tempi %>%
  select(T, m, rapporto_migliore_expar2 = rapporto_migliore) %>%
  left_join(
    riepilogo_tempi_expar1 %>%
      select(T, m, rapporto_migliore_expar1 = rapporto_migliore),
    by = c("T", "m")
  )

print(confronto_expar1_expar2)

# Interpretazione: se i due rapporti sono simili per ogni combinazione (T, m),
# il vantaggio computazionale dell'ELM rispetto alla NNET non dipende in modo
# sostanziale dal vero ordine del processo generatore (EXPAR(1) vs EXPAR(2)),
# ma solo dalla dimensione dello strato nascosto e dalla lunghezza della
# serie - coerentemente con quanto riportato nel paper originale.

## -----------------------------------------------------------------------
## 12.7 Barplot di confronto: rapporto_migliore per EXPAR(1) vs EXPAR(2)
## -----------------------------------------------------------------------
# Rappresenta graficamente il confronto gia' calcolato al punto 14.6:
# per ciascuna combinazione (T, m) si affiancano due barre, una per il
# rapporto_migliore dell'EXPAR(1) (p = 1) e una per quello dell'EXPAR(2)
# (p = p_fisso = 2), cioe' il rapporto tra il tempo del singolo NNET
# migliore e il tempo di stima dell'ELM.

# 12.7.1 si trasforma confronto_expar1_expar2 dal formato "largo" (una
# colonna per modello) al formato "lungo" (una riga per combinazione
# T-m-modello), necessario per affiancare le barre con ggplot2
dati_barplot_confronto <- confronto_expar1_expar2 %>%
  tidyr::pivot_longer(
    cols = c(rapporto_migliore_expar1, rapporto_migliore_expar2),
    names_to = "modello",
    values_to = "rapporto_migliore"
  ) %>%
  mutate(
    modello = recode(
      modello,
      rapporto_migliore_expar1 = "EXPAR(1)",
      rapporto_migliore_expar2 = "EXPAR(2)"
    )
  )

# 12.7.2 barplot con barre affiancate (una per modello), facet per m,
# asse x = lunghezza della serie T
grafico_confronto_expar1_expar2 <- ggplot(
  dati_barplot_confronto,
  aes(x = factor(T), y = rapporto_migliore, fill = modello)
) +
  geom_col(position = position_dodge()) +
  facet_wrap(~ m, nrow = 1, labeller = label_both) +
  labs(
    title = "Confronto EXPAR(1) vs EXPAR(2): rapporto tempo NNET migliore / tempo ELM",
    subtitle = "Per ciascuna dimensione m dello strato nascosto e lunghezza T della serie",
    x = "Lunghezza della serie storica (T)",
    y = "Rapporto tempo di calcolo NNET / ELM",
    fill = "Modello generatore dei dati"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(grafico_confronto_expar1_expar2)

# Interpretazione: barre di altezza simile tra i due modelli, per ogni
# combinazione (T, m), indicano che il vantaggio computazionale dell'ELM
# rispetto al singolo NNET migliore non dipende in modo sostanziale dal vero
# ordine del processo generatore dei dati (EXPAR(1) vs EXPAR(2)); differenze
# sistematiche indicherebbero invece una sensibilita' del vantaggio
# computazionale alla complessita' del modello generatore.