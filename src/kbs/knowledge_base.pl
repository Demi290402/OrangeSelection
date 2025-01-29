% Knowledge Base aggiornata per AIProlog

% Fatti: Qualità delle varietà
fact(qualita(valencia, alta)).
fact(qualita(navel, media)).
fact(qualita(cara_cara, alta)).
fact(qualita(blood_orange, alta)).
fact(qualita(hamlin, bassa)).

% Fatti: Proprietà chimiche
fact(dolcezza(valencia, 12)).
fact(dolcezza(navel, 8)).
fact(dolcezza(cara_cara, 11)).
fact(dolcezza(blood_orange, 13)).
fact(dolcezza(hamlin, 6)).

fact(acidita(valencia, 3.2)).
fact(acidita(navel, 3.4)).
fact(acidita(cara_cara, 3.0)).
fact(acidita(blood_orange, 3.8)).
fact(acidita(hamlin, 3.6)).

% Fatti: Origini delle varietà
fact(origine(valencia, spagna)).
fact(origine(navel, california)).
fact(origine(cara_cara, venezuela)).
fact(origine(blood_orange, sicilia)).
fact(origine(hamlin, florida)).

% Regola: Qualità basata sulla dolcezza
rule(qualita_dolce(Varieta, alta),
     [fact(dolcezza(Varieta, Dolcezza)), greater_than(Dolcezza, 10)]).

rule(qualita_dolce(Varieta, bassa),
     [fact(dolcezza(Varieta, Dolcezza)), less_or_equal(Dolcezza, 10)]).

% Regola: Raccomandare varietà di alta qualità
rule(raccomanda(Varieta),
     [fact(qualita(Varieta, alta)),
      fact(dolcezza(Varieta, Dolcezza)), greater_than(Dolcezza, 10),
      fact(acidita(Varieta, Acidita)), less_or_equal(Acidita, 3.5)]).

% Regola avanzata: Analisi combinata di dolcezza e acidità
rule(analisi_combinata(Varieta, "Ottima qualità e bilanciamento"),
     [fact(dolcezza(Varieta, Dolcezza)), greater_than(Dolcezza, 12),
      fact(acidita(Varieta, Acidita)), less_than(Acidita, 3.0)]).

% Collegamenti al Web Semantico
% RDF: Rappresentazione delle varietà come entità del web semantico
rdf(arancia_valencia, 'http://example.org/arance#Valencia').
rdf(arancia_navel, 'http://example.org/arance#Navel').
rdf(arancia_cara_cara, 'http://example.org/arance#CaraCara').
rdf(arancia_blood_orange, 'http://example.org/arance#BloodOrange').
rdf(arancia_hamlin, 'http://example.org/arance#Hamlin').

% Regole: Recupero di dati RDF
query_rdf(Varieta, URI) :- rdf(Entity, URI), atom_concat('arancia_', Varieta, Entity).

% Regole: Raccomandazione basata su proprietà combinate
raccomanda(Varieta) :-
    qualita(Varieta, alta),
    dolcezza(Varieta, Dolcezza), Dolcezza > 10,
    acidita(Varieta, Acidita), Acidita =< 3.5.

% Regola 1: Filtra varietà di alta qualità
rule(varieta_alta_qualita(Varieta), 
     [fact(qualita(Varieta, alta))]).

% Regola 2: Filtra varietà dolci con dolcezza > 10
rule(varieta_dolce(Varieta),
     [fact(dolcezza(Varieta, Dolcezza)), greater_than(Dolcezza, 10)]).

% Regola 3: Trova varietà con origine specifica
rule(varieta_con_origine(Varieta, Origine), 
     [fact(origine(Varieta, Origine))]).

% Ragionamento concatenato: Trova varietà di alta qualità, dolci e con origine specifica
rule(varieta_qualita_dolce_con_origine(Varieta, Origine), 
     [rule(varieta_alta_qualita(Varieta), _),
      rule(varieta_dolce(Varieta), _),
      rule(varieta_con_origine(Varieta, Origine), _)]).
