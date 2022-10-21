background_claim-GOLD -> own_claim
==================================
A32[1787:4935]
1. hard (factual statement w/o ref)
2. hard (describes content of other work, but looks like an author statement)
3. easy (followed by ref)

A32[25261:27768]
1. medium (no ref, but intruduced with "Typically,")

A32[4935:7737]
1. very hard (no ref, surrounded by a lot of own_claims)

A33[14720:20675]
1. medium (after ref, but ref is cut and content before interrupted with figure descriptions)
2. easy (after ref, introduced with "who noted")

A33[1689:5602]
1. very hard (annotation error?)
2. medium (no ref, but "Over the course of roughly half a dozen papers on [...] a preveiling strategy has emerged.", a bit strange)
3. easy-medium (no ref, but introduced by data ADU)
4. medium (no ref, followed by own_claim)
5. medium (no ref, followed by own_claim)
6. easy (before ref)
7. hard (no ref, introduced with "In fact", followed by own_claim)

A33[20675:26705]
1. very hard (annotation correct? probably, but strange)
2. very hard (probably annotation error)
3. medium (after ref, but it is just a part of the adu which starts already before the ref)
4. hard (is about a concept introduced, with ref, in another section - content not accessible here)
5. very hard (annotation error? surrounded by own_claim)

A33[26705:31336]
1. hard (annotation error?, after ref, but surrounded by own_claim)

A34[10054:24577]
1. hard (no ref, pronoun "we" as actor)
2. hard (no ref, "us")
3. easy-medium (mathematical reformulation after background_claim "the law of [...] says"). 
note: this is followed by at least two annotation errors that are "fixed" by the model predictions follow`

A34[5762:10054]
1. medium (no ref, followed by own_claim)
2. medium (no ref, followed by own_claim)
3. easy (no ref, surrounded by own_claim)
4. medium (no ref, but followed by data+background_claim)
5. easy-medium (following own_claim, but followed by multiple refs, but content after is distorted)

A35[11796:32248]
1. hard (annotation error?, no refs, surrounded by own_claims)
2. hard (annotation error?, no refs, surrounded by own_claims)
3. medium-hard (no refs, general statement subtly marked by language, surrounded by non-ADUs and own claims (but not linked via relations))
4. medium-hard (no refs, general statement subtly marked by language, surrounded by non-ADUs and own claims (but not linked via relations))
5. medium (no refs, surrounded by other background_claims, but as introduction to own_claims (but just linked internally, not to that)) 
6. medium (no refs, surrounded by other background_claims, but as introduction to own_claims (but just linked internally, not to that))
7. medium (no refs, surrounded by background_claims, but language sounds a bit "claimish") NOTE: before these errors are interesting examples for prediction "errors" that may be better than the annotations (introductory phrases removed from claim; implications as extra ADUs)
8. easy-medium (no refs, but surrounded by background_claims)

A35[4032:11796]
1. medium (no refs, in method section, sequence of a lot of background_claims, but all not detected, heavily linked via support relations)
2. medium (no refs, in method section, sequence of a lot of background_claims, but all not detected, heavily linked via support relations)
3. medium (no refs, in method section, sequence of a lot of background_claims, but all not detected, heavily linked via support relations)
4. medium (no refs, but general description of a method, language a bit "claimish", surrounded by own_claims) Note: followed by example of ADU that is "erroneously" split into pieces (but wrong type because of "exaggerative" language)
5. medium (no refs, in between background and own_claim, but "backgroundish" language as part of this ADU: "This method is supposed to [...] that are not subject to [...]", not linked via relation)
6. medium (no refs, embedded in own_claims, but "backgroundish" language as part of this ADU: "It is supposed to [...]")
7. medium (after ref, but just part (linked with parts_of_same), nested in own_claims)
8. medium (after ref, but just part (linked with parts_of_same), nested in own_claims)
9. medium-hard (no refs, just part (linked with parts_of_same), in method section (but just followed with one own_claim)) NOTE: errors show interesting additional data ADUs and "fix" of erroneously (?) split ADU 
10. medium (no refs, very "claimish" (superlative), followed by own_claim)

A36[45769:48083]
1. medium-hard (no refs, surrounded by a lot of own_claims, but language marks (universal quantifier)) NOTE: a lot of correctly predicted ADUs here

A36[5494:9463]
1. medium (no refs, between background and own_claim, but general statement (marked by language via "any")) NOTE: also a lot of correct predictions

A36[9463:20338] NOTE: also a lot of correct predictions
1. easy-medium (surrounded by refs, but own_claim before confusing (same pattern with respect to ref) and distorted text content)
2. medium (parts_of_same around ref and in between own_claim)
3. medium (parts_of_same around ref and in between own_claim)

A37[11509:25046]
1. easy-medium (no ref, but surrounded by background_claims) NOTE: line 20: ADU is correctly (?) split up
2. easy (before refs marked and detected as data, but these are only linked to correctly predicted background_claim after these) NOTE: a lot of equations that are predicted as own_claim, but are no gold ADU at all

A37[1452:4213]
1. hard (annotation error? followed by ref, but ADU just states about own work, i.e. that they use the method from the ref)
2. medium (no ref, but general statement, after own_claim)
3. medium (with ref, but parts_of_same) NOTE: it looks like the ref is misplaced here
4. easy-medium (no ref, but general statement, not linked)

A37[28550:31536]
1. hard (with ref, but parts_of_same)
2. medium-hard (ref, but linked to other ADU with "contradicts")

A38[14489:20317]
1. medium-hard (no ref, surrounded by own_claim, parts_of_same around data)

A38[45461:49432]
1. medium-hard (no ref, after many own_claims, parts_of_same around discourse marker "however")
2. medium (no ref, surrounded by a lot of main_claims) NOTE: a lot of interesting "hallucinated" ADUs here 

A38[8529:14489]
1. easy (after ref) NOTE: linking seems to be wrong here (the own_claim should be supported by the background_claim und not by the data, i.e. the ref), but also ref in text is misplaced (should be after author names, or what is the number?)
2. medium (no direct refs, follow-up error)
3. medium-hard (no direct refs, follow-up error, parts_of_same, very "claimish")
4. medium-hard (no direct refs, follow-up error, parts_of_same, very "claimish")
5. medium (before ref, but parts_of_same)
6. medium (no ref, linked via contradicts, very "claimish")
7. medium (no ref, general statement)
8. medium (no ref, general statement)
9. medium (after ref, but parts_of_same)

A39[14015:21965] NOTE: good prediction quality
1. easy-medium (no refs, general statement introduces with "A common practice [...]", but surrounded by own_claim)
2. medium (no refs, general statement, but surrounded by own_claim)

A39[1825:14015]
1. medium (no refs, switch from own to background_claim, but general statement introduced (before the ADU) with "The reader should know that [...]") NOTE: a lot of correct predictions
2. hard (no refs, switch from own to background_claim, parts_of_same, introduced with author mention "[...] what is _our_ main concern [...]" )
3. medium (no refs, single ADU surrounded by own_claim, but general statement containing marker word "[...] do not _in general_ solve [...]")
4. hard (refs quite far before, contains very misleading phrase "[...] in this work")

A39[25488:37632]
1. medium (no refs, general statement, parts_of_same, surrounded by own_claim)
2. medium (no refs, general statement, parts_of_same, surrounded by own_claim)

A39[37632:54129]
1. medium (no refs, general statement, surrounded by own_claim)
2. medium (no refs, general statement, surrounded by own_claim)
3. easy-medium (no refs, general statement with marker "[...] will _generally_ not enforce [...]", surrounded by own_claim)
4. medium-hard (no refs, general statement, surrounded by own_claim, parts_of_same)
5. medium (after ref, before own_claim, parts_of_same)
6. medium (no refs, general statement with marker "[...] are difficult _in general_", surrounded by own_claim)

A39[54129:61217]
1. easy (after ref and introduced with "[ref], which [...]", but surrounded by own_claims)
2. easy (no ref, surrounded by own_claim, general statement marked with switch from past tense to present tense)

A39[61217:63397]
1. hard (no ref, general statement marked by plural "[...] these methods tend [...]", surrounded by own_claim)


A40[9215:23372]
1. medium (no refs, chain of background_claims rooted in a common sense argument)
2. medium (no refs, general statement)
3. medium (surrounded by background_claims, but sounds very "claimish")
4. hard (no refs, between background and own_claim, reference to author "It enables us to model [...]" (like in following own_claim "We present a model [...]")
5. hard (no refs, intermixed with own_claims that sound very similar, missed switch to background_claim )
6. hard (no refs, follow-up)
7. hard (no refs, follow-up, parts_of_same)
8. hard (no refs, follow-up, contradicts, after missing data annotations?)
9. hard (no refs, follow-up, followed by own_claim)
10. medium (no refs, after own_claim, general statement)
11. medium-hard (no refs, mention of author "[A] drawback of this model for _our_ application is [...]"!)
12. medium-hard (no refs, follow-up, followed by own_claim)
13. medium (no refs, embedded in own_claim) NOTE: good example for this error 
14. medium (no refs, follow-up)
15. medium (no refs, follow-up, contradicts)
16. medium (no refs, follow-up, contradicts)


TODOS:
* Per input file evaluation, i.e. group files by original file names, e.g. A01, A02 etc.
* Scores for ADU attachment, i.e. treat all labels as one. Can be calculated from confusion matrices.