# -*- coding: UTF-8 -*-
"""Language specific data and functions."""

from __future__ import unicode_literals
try:
	import re2 as re
except ImportError:
	import re
import collections

VOWELS = 'aoeuiäàâáåãëéèêóòöôõðùúüìíïî'  # y is special case; true for en.
VOWELS_FR = VOWELS + 'yÿ'

specialsyllables_en = """\
tottered 2
chummed 1
peeped 1
moustaches 2
shamefully 3
messieurs 2
satiated 4
sailmaker 4
sheered 1
disinterred 3
propitiatory 6
bepatched 2
particularized 5
caressed 2
trespassed 2
sepulchre 3
flapped 1
hemispheres 3
pencilled 2
motioned 2
poleman 2
slandered 2
sombre 2
etc 4
sidespring 2
mimes 1
effaces 2
mr 2
mrs 2
ms 1
dr 2
st 1
sr 2
jr 2
truckle 2
foamed 1
fringed 2
clattered 2
capered 2
mangroves 2
suavely 2
reclined 2
brutes 1
effaced 2
quivered 2
h'm 1
veriest 3
sententiously 4
deafened 2
manoeuvred 3
unstained 2
gaped 1
stammered 2
shivered 2
discoloured 3
gravesend 2
60 2
lb 1
unexpressed 3
greyish 2
unostentatious 5
"""

fallback_cache = {}
_fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou",
		"sia$", ".ely$"]
_fallback_addsyl = ["ia", "riet", "dien", "iu", "io", "ii",
		"[aeiouy]bl$", "mbl$",
		"[aeiou]{3}",
		"^mc", "ism$",
		"(.)(?!\\1)([aeiouy])\\2l$",
		"[^l]llien",
		"^coad.", "^coag.", "^coal.", "^coax.",
		"(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
		"dnt$"]
fallback_subsyl = [re.compile(a) for a in _fallback_subsyl]
fallback_addsyl = [re.compile(a) for a in _fallback_addsyl]


def _normalize_word(word):
	return word.strip().lower()

# Read syllable overrides and populate cache with them
for line in specialsyllables_en.splitlines():
	line = line.strip()
	if line:
		toks = line.split()
		assert len(toks) == 2
		fallback_cache[_normalize_word(toks[0])] = int(toks[1])


def countsyllables_en(word):
	"""Fallback syllable counter.

	This is based on the algorithm in Greg Fast's perl module
	Lingua::EN::Syllable."""
	if not word:
		return 0

	# Remove final silent 'e'
	if word[-1] == "e":
		word = word[:-1]

	# Check for a cached syllable count
	if word in fallback_cache:
		return fallback_cache[word]

	# Count vowel groups
	result = 0
	prev_was_vowel = False
	for char in word:
		is_vowel = char in VOWELS or char == 'y'
		if is_vowel and not prev_was_vowel:
			result += 1
		prev_was_vowel = is_vowel

	# Add & subtract syllables
	for r in fallback_addsyl:
		if r.search(word):
			result += 1
	for r in fallback_subsyl:
		if r.search(word):
			result -= 1

	# Cache the syllable count
	fallback_cache[word] = result

	return result


def countsyllables_nlde(word):
	"""Count syllables for Dutch / German words by counting vowel-consonant or
	consonant-vowel pairs, depending on the first character being a vowel or
	not. If it is, a trailing e will be handled with a special rule."""
	result = 0
	prev_was_vowel = word[0] in VOWELS
	for char in word[1:]:
		is_vowel = char in VOWELS
		if prev_was_vowel and not is_vowel:
			result += 1
		prev_was_vowel = is_vowel

	if (len(word) > 1 and word[0] in VOWELS
			and word.endswith('e') and not word[-2] in VOWELS):
		result += 1
	return result or 1

# Using Pyphen hyphenation to count french syllables. (https://pyphen.org/)
def count_syllables_fr(word):
	import pyphen
	dic = pyphen.Pyphen(lang='fr')
	# Count the syllables as the number of hyphenated parts (minimum 1):
	return max(1, len(dic.inserted(word).split('-')))


conjuction_en = r'and|but|or|yet|nor'
preposition_en = (
		'board|about|above|according to|across from'
		'|after|against|alongside|alongside of|along with'
		'|amid|among|apart from|around|aside from|at|away from'
		'|back of|because of|before|behind|below|beneath|beside'
		'|besides|between|beyond|but|by means of'
		'|concerning|considering|despite|down|down from|during'
		'|except|except for|excepting for|from among'
		'|from between|from under|in addition to|in behalf of'
		'|in front of|in place of|in regard to|inside of|inside'
		'|in spite of|instead of|into|like|near to|off'
		'|on account of|on behalf of|onto|on top of|on|opposite'
		'|out of|out|outside|outside of|over to|over|owing to'
		'|past|prior to|regarding|round about|round'
		'|since|subsequent to|together|with|throughout|through'
		'|till|toward|under|underneath|until|unto|up'
		'|up to|upon|with|within|without|across|along'
		'|by|of|in|to|near|of|from')
pronoun_en = (
		'i|me|we|us|you|he|him|she|her|it|they'
		'|them|thou|thee|ye|myself|yourself|himself'
		'|herself|itself|ourselves|yourselves|themselves'
		'|oneself|my|mine|his|hers|yours|ours|theirs|its'
		'|our|that|their|these|this|those|your')
words_en = collections.OrderedDict([
	('tobeverb', re.compile(
		r'\b(be|being|was|were|been|are|is)\b', re.IGNORECASE)),
	('auxverb', re.compile(
		r"\b(will|shall|cannot|may|need to|would|should"
		r"|could|might|must|ought|ought to|can't|can)\b", re.IGNORECASE)),
	('conjunction', re.compile(
		'\\b(%s)\\b' % conjuction_en, re.IGNORECASE)),
	('pronoun', re.compile(
		'\\b(%s)\\b' % pronoun_en, re.IGNORECASE)),
	('preposition', re.compile(
		'\\b(%s)\\b' % preposition_en, re.IGNORECASE)),
	# a bit limited, but this is exactly what the original style(1) did:
	('nominalization', re.compile(
		r'\b\w{3,}(tion|ment|ence|ance)\b', re.IGNORECASE | re.UNICODE)),
	])

beginnings_en = collections.OrderedDict([
	('pronoun', re.compile(
		'(^|\\n)(%s)\\b' % pronoun_en, re.IGNORECASE)),
	('interrogative', re.compile(
		r'(^|\n)(why|who|what|whom|when|where|how)\b', re.IGNORECASE)),
	('article', re.compile(
		r'(^|\n)(the|a|an)\b', re.IGNORECASE)),
	('subordination', re.compile(
		r"(^|\n)(after|because|lest|till|'til|although"
		r"|before|now that|unless|as|even if|provided that|provided"
		r"|until|as if|even though|since|as long as|so that"
		r"|whenever|as much as|if|than|as soon as|inasmuch"
		r"|in order that|though|while)\b", re.IGNORECASE)),
	('conjunction', re.compile(
		'(^|\\n)(%s)\\b' % conjuction_en, re.IGNORECASE)),
	('preposition', re.compile(
		'(^|\\n)(%s)\\b' % preposition_en, re.IGNORECASE)),
	])

conjuction_nl = 'en|maar|of|want|dus|noch'
preposition_nl = (
		"à|aan|ad|achter|behalve|beneden|betreffende|bij"
		"|binnen|blijkens|boven|buiten|circa|conform|contra"
		"|cum|dankzij|door|gedurende|gezien|hangende|in"
		"|ingevolge|inzake|jegens|krachtens|langs|met|middels"
		"|mits|na|naar|naast|nabij|namens|niettegenstaande"
		"|nopens|om|omstreeks|omtrent|ondanks|onder|ongeacht"
		"|onverminderd|op|over|overeenkomstig|per|plus|richting"
		"|qua|rond|rondom|sedert|staande|te|tegen|tegenover"
		"|ten|ter|tijdens|tot|tussen|uit|uitgezonderd|van"
		"|vanaf|vanuit|vanwege|versus|via|volgens|voor"
		"|voorbij|wegens|zonder")
pronoun_nl = (
		# persoonlijk voornaamwoord
		"ik|jij|je|u|hij|hem|zij|ze|haar|het"
		"|wij|we|ons|jullie|hen|hun"
		# wederkerend voornaamwoord
		"mij|me|mijzelf|mezelf|je|jezelf|uzelf"
		"|zich|zichzelf|haarzelf|onszelf"
		"|elkaar|elkaars|elkander|elkanders|mekaar|mekaars"
		# pers. vnw: archaisch
		"gij|ge"
		"|mijnen|deinen|zijnen|haren|onzen|uwen|hunnen|haren"
		"|mijner|deiner|zijner|harer|onzer|uwer|hunner|harer"
		"|mijnes|deines|zijnes|hares|onzes|uwes|hunnes|hares")
words_nl = collections.OrderedDict([
	('tobeverb', re.compile(
		r'\b(ben|bent|is|zijn|was|waren)\b', re.IGNORECASE)),
	('auxverb', re.compile(
		"\\b("
		# NB: past perfect forms of these verbs
		# ('gehad', 'geweest', 'geworden') are not auxiliary.
		# with past perfect verb
		"heb|hebt|heeft|hebben|had|hadden"
		"|word|wordt|worden|werd|werden"
		# "|ben|bent|is|zijn|was|waren"
		# with infinitive
		"|zal|zult|zullen|zou|zouden"
		"|kan|kan|kunt|kunnen|kon|konden"
		"|wil|wilt|willen|wilde|wilden|wou|wouden"
		"|moet|moeten|moest|moesten"
		# "|mag|mogen|mocht|mochten"
		# "|hoef|hoeft|hoeven|hoefde|hoefden"
		# "|doe|doet|doen|deed|deden"
		")\\b", re.IGNORECASE)),
	('conjunction', re.compile(
		'\\b(%s)\\b' % conjuction_nl, re.IGNORECASE)),
	('pronoun', re.compile(
		'\\b(%s)\\b' % pronoun_nl, re.IGNORECASE)),
	('preposition', re.compile(
		'\\b(%s)\\b' % preposition_nl, re.IGNORECASE)),
	# a bit limited, but this is exactly what the original style(1) did:
	('nominalization', re.compile(
		r'\b.{3,}(tie|heid|ing|end|ende)\b', re.IGNORECASE)),
	])
beginnings_nl = collections.OrderedDict([
	('pronoun', re.compile(
		'(^|\\n)(%s)\\b' % pronoun_nl, re.IGNORECASE)),
	('interrogative', re.compile(
		r'(^|\n)(wie|wat|waar|waarom|wanneer|hoe|welk|welke)\b',
		re.IGNORECASE)),
	('article', re.compile(
		r"(^|\n)(de|het|een|'t)\b", re.IGNORECASE)),
	('subordination', re.compile(
		"(^|\\n)("
		# onderschikkende voegwoorden
		"aangezien|als|alsof|behalve|daar|daarom|dat"
		"|derhalve|doch|doordat|hoewel|indien|mits|nadat"
		"|noch|ofschoon|omdat|ondanks|opdat|sedert|sinds"
		"|tenzij|terwijl|toen|totdat|voordat|wanneer"
		"|zoals|zodat|zodra|zonder dat"
		# infitief constructies
		"|om te)\\b", re.IGNORECASE)),
	('conjunction', re.compile(
		'(^|\\n)(%s)\\b' % conjuction_nl, re.IGNORECASE)),
	('preposition', re.compile(
		'(^|\\n)(%s)\\b' % preposition_nl, re.IGNORECASE)),
	])

conjuction_de = ('und|oder|aber|sondern|doch|nur|bloß|denn'
		'weder|noch|sowie')
preposition_de = (
	'aus|außer|bei|mit|nach|seit|von|zu'
	'|bis|durch|für|gegen|ohne|um|an|auf'
	'|hinter|in|neben|über|unter|vor|zwischen'
	'|anstatt|statt|trotz|während|wegen')
pronoun_de = (
	'ich|du|er|sie|es|wir|ihr'  # sie           # Nominativ
	'|mich|dich|ihn|uns|euch'  # sie             # Akkusativ
	'|mir|dir|ihm|ihnen'  # uns euch ihr         # Dativ
	'|mein|dein|sein|unser|euer'  # ihr          # Genitiv
	'|meiner|deiner|seiner|unserer|eurer|ihrer'  # Genitiv
	'|meine|deine|seine|unsere|eure|ihre'        # Genitiv
	'|meines|deines|seines|unseres|eures|ihres'  # Genitiv
	'|meinem|deinem|seinem|unserem|eurem|ihrem'  # Genitiv
	'|meinen|deinen|seinen|unseren|euren|ihren'  # Genitiv
		)
words_de = collections.OrderedDict([
	('tobeverb', re.compile("\\b("
		"sein|bin|bist|ist|sind|seid|war|warst|wart"
		"|waren|gewesen|wäre|wärst|wär|wären|wärt|wäret"
		")\\b", re.IGNORECASE)),
	('auxverb', re.compile("\\b("
		"haben|habe|hast|hat|habt|gehabt|hätte|hättest"
		"|hätten|hättet"
		"|werden|werde|wirst|wird|werdet|geworden|würde"
		"|würdest|würden|würdet"
		"|können|kann|kannst|könnt|konnte|konntest|konnten"
		"|konntet|gekonnt|könnte|könntest|könnten|könntet"
		"|müssen|muss|muß|musst|müsst|musste|musstest|mussten"
		"|gemusst|müsste|müsstest|müssten|müsstet"
		"|sollen|soll|sollst|sollt|sollte|solltest|solltet"
		"|sollten|gesollt"
		")\\b", re.IGNORECASE)),
	('conjunction', re.compile(
		'\\b(%s)\\b' % conjuction_de, re.IGNORECASE)),
	('pronoun', re.compile(
		'\\b(%s)\\b' % pronoun_de, re.IGNORECASE)),
	('preposition', re.compile(
		'\\b(%s)\\b' % preposition_de, re.IGNORECASE)),
	('nominalization', re.compile(
		r'\b.{3,}(ung|heit|keit|nis|tum)\b', re.IGNORECASE)),
	])
beginnings_de = collections.OrderedDict([
	('pronoun', re.compile(
		'(^|\\n)(%s)\\b' % pronoun_de, re.IGNORECASE)),
	('interrogative', re.compile(
		r'(^|\n)(wer|was|wem|wen|wessen|wo|wie|warum|weshalb|wann'
		r'|wieso|weswegen)\b', re.IGNORECASE)),
	('article', re.compile(
		r"(^|\n)(der|die|das|des|dem|den|ein|eine|einer|eines|einem|einen)\b",
		re.IGNORECASE)),
	('subordination', re.compile("(^|\\n)("
		# bei Nebensätzen
		"als|als dass|als daß|als ob|anstatt dass|anstatt daß"
		"|ausser dass|ausser daß|ausser wenn|bevor|bis|da|damit"
		"|dass|daß|ehe|falls|indem|je|nachdem|ob|obgleich"
		"|obschon|obwohl|ohne dass|ohne daß|seit|so daß|sodass"
		"|sobald|sofern|solange|so oft|statt dass|statt daß"
		"|während|weil|wenn|wenn auch|wenngleich|wie|wie wenn"
		"|wiewohl|wobei|wohingegen|zumal"
		# bei Infinitivgruppen
		"|als zu|anstatt zu|ausser zu|ohne zu|statt zu|um zu"
		")\\b", re.IGNORECASE)),
	('conjunction', re.compile(
		'(^|\\n)(%s)\\b' % conjuction_de, re.IGNORECASE)),
	('preposition', re.compile(
		'(^|\\n)(%s)\\b' % preposition_de, re.IGNORECASE)),
	])

################################################################################
# French coordinating conjunctions
# Source: Projet Voltaire – https://www.projet-voltaire.fr/dossier-voltaire/conjonctions-de-coordination/
conjunction_fr = (
	"mais|ou|et|donc|or|ni|car"
)

# French prepositions
# Source: Lexique.org / Lexique383, sorted by 'freqlivres' (frequency in books).
preposition_fr = (
    "de|à|d'|en|dans|pour|sur|avec|par|sans"
    "|sous|vers|entre|après|devant|chez|contre|avant|depuis|pendant"
    "|derrière|voilà|près|dès|malgré|parmi|selon|voici|auprès|hors"
    "|sauf|grâce|par-dessus|durant|plein|afin de|jusque|envers|vis-à-vis"
    "|suivant|outre|concernant|par-delà|à l'instar|par-derrière|hormis"
	"|via|moyennant|excepté|revoilà|nonobstant|devers|par-dessous"
    "|par-devant|à l'encontre|revoici|par-devers|ès|dudit|jusqu'à"
    "|audit|desdits|auxdits|desdites|jusqu'au"
)

# French pronouns (personal, demonstrative, indefinite, relative, reflexive)
# Source: Lexique.org / Lexique383, sorted by 'freqlivres' (frequency in books).
pronoun_fr = (
    "il|je|qui|elle|se|s'|on|me|lui|nous"
    "|vous|y|que|ce|ils|l'|tu|ça|en|le"
    "|moi|où|rien|les|la|dont|tout|t'|te|cela"
    "|autre|elles|eux|toi|leur|quoi|autres|un|celui|personne"
    "|ceux|celle|tous|l'un|lui-même|une|chacun|laquelle|lequel|quelqu'un"
    "|elle-même|celles|moi-même|l'une|d'autres|soi|celui-ci|uns|celle-ci|celui-là"
    "|même|ceci|aucun|eux-mêmes|auquel|lesquels|lesquelles|mienne|sienne"
    "|mien|sien|chacune|certains|nul|siens|celle-là|aucune|ceux-là|soi-même"
    "|toutes|auxquels|nôtre|la plupart|duquel|quelques-uns|nôtres|miens|ceux-ci"
    "|auxquelles|unes|mêmes|vous-même|tel|nous-mêmes|elles-mêmes|plusieurs"
    "|autrui|toi-même|vôtre"
)
# All forms of "to be" for French.
# Source : Bescherelle https://conjugaison.bescherelle.com/verbes/etre
tobe_verb_fr = (
    "suis|es|est|sommes|êtes|sont"
    "|étais|était|étions|étiez|étaient"
    "|fus|fut|fûmes|fûtes|furent"
    "|serai|seras|sera|serons|serez|seront"
    "|serais|serait|serions|seriez|seraient"
    "|sois|soit|soyons|soyez|soient"
    "|fusse|fusses|fût|fussions|fussiez|fussent"
    "|être|étant|été"
)

# French modal auxiliary verbs (pouvoir, devoir, vouloir, savoir) + falloir
# Source : https://leconjugueur.lefigaro.fr/frlesauxiliaires.php
auxverb_fr = (
    "pouvoir|peux|peut|pouvons|pouvez|peuvent"
	"|pouvais|pouvait|pouvions|pouviez|pouvaient"
    "|devoir|dois|doit|devons|devez|doivent"
	"|devais|devait|devions|deviez|devaient"
    "|vouloir|veux|veut|voulons|voulez|veulent"
	"|voulais|voulait|voulions|vouliez|voulaient"
    "|savoir|sais|sait|savons|savez|savent"
	"|savais|savait|savions|saviez|savaient"
    "|falloir|faut|faudra|faudrait|fallait|fallu"
)

# French subordinating conjunctions
# Source: Projet Voltaire – https://www.projet-voltaire.fr/dossier-voltaire/conjonctions-de-subordination/
subordination_fr = (
	"comme|lorsque|puisque|quand|si"
	"|à condition que|à mesure que|ainsi que|au cas où|d’autant que"
    "|pour que|afin que|c’est pourquoi|de sorte que|de peur que"
	"|parce\s+qu[’'e]?|étant donné que|du fait que"
    "|alors que|bien que|quoique|encore que|qui que"
    "|avant que|aussitôt que|tant que"
	"|de même que|plus que|moins que"
	"|à supposer que|dans la mesure où"
	"|sans que|selon que"
	"|au point que|tellement que|que"
)

# French articles
# Source: Lexique.org / Lexique383
article_fr = (
    "le|la|les|l'|un|une|des|du|au|aux"
)

words_fr = collections.OrderedDict([
	('tobeverb', re.compile(
		'\\b(%s)\\b' % tobe_verb_fr, re.IGNORECASE)),
	('auxverb', re.compile(
		'\\b(%s)\\b' % auxverb_fr, re.IGNORECASE)),
	('conjunction', re.compile(
		'\\b(%s)\\b' % conjunction_fr, re.IGNORECASE)),
	('preposition', re.compile(
		'\\b(%s)\\b' % preposition_fr, re.IGNORECASE)),
	('nominalization', re.compile(
		r'\b\w{2,}(tion|sion|ment|ence|ance|age|ure|ité|té|eur|euse|isme)\b', re.IGNORECASE | re.UNICODE)),
	('subordination', re.compile(
		'\\b(%s)\\b' % subordination_fr, re.IGNORECASE)),
	('article', re.compile(
		'\\b(%s)\\b' % article_fr, re.IGNORECASE)),
])

beginnings_fr = collections.OrderedDict([
	('pronoun', re.compile(
		'(^|\\n)(%s)\\b' % pronoun_fr, re.IGNORECASE)),
	('interrogative', re.compile(
		r'(^|\n)(pourquoi|qui|que|quoi|quand|où|comment)\b', re.IGNORECASE)),
])
################################################################################
# Long Dale-Chall word list of 3000 words recognized by 80 % of fifth graders
basicwords_en = frozenset("""
n't 'm 'll 'd 's 're 've
t m ll d s re ve don shouldn aren didn hadn hasn haven isn needn shan wasn
a able aboard about above absent accept accident account
ache aching acorn acre across act acts add address admire adventure afar afraid
after afternoon afterward afterwards again against age aged ago agree ah ahead
aid aim air airfield airplane airport airship airy alarm alike alive all alley
alligator allow almost alone along aloud already also always am America
American among amount an and angel anger angry animal another answer ant any
anybody anyhow anyone anything anyway anywhere apart apartment ape apiece
appear apple April apron are aren't arise arithmetic arm armful army arose
around arrange arrive arrived arrow art artist as ash ashes aside ask asleep at
ate attack attend attention August aunt author auto automobile autumn avenue
awake awaken away awful awfully awhile ax axe baa babe babies back background
backward backwards bacon bad badge badly bag bake baker bakery baking ball
balloon banana band bandage bang banjo bank banker bar barber bare barefoot
barely bark barn barrel base baseball basement basket bat batch bath bathe
bathing bathroom bathtub battle battleship bay be beach bead beam bean bear
beard beast beat beating beautiful beautify beauty became because become
becoming bed bedbug bedroom bedspread bedtime bee beech beef beefsteak beehive
been beer beet before beg began beggar begged begin beginning begun behave
behind being believe bell belong below belt bench bend beneath bent berries
berry beside besides best bet better between bib bible bicycle bid big bigger
bill billboard bin bind bird birth birthday biscuit bit bite biting bitter
black blackberry blackbird blackboard blackness blacksmith blame blank blanket
blast blaze bleed bless blessing blew blind blindfold blinds block blood bloom
blossom blot blow blue blueberry bluebird blush board boast boat bob bobwhite
bodies body boil boiler bold bone bonnet boo book bookcase bookkeeper boom boot
born borrow boss both bother bottle bottom bought bounce bow bowl bow-wow box
boxcar boxer boxes boy boyhood bracelet brain brake bran branch brass brave
bread break breakfast breast breath breathe breeze brick bride bridge bright
brightness bring broad broadcast broke broken brook broom brother brought brown
brush bubble bucket buckle bud buffalo bug buggy build building built bulb bull
bullet bum bumblebee bump bun bunch bundle bunny burn burst bury bus bush
bushel business busy but butcher butt butter buttercup butterfly buttermilk
butterscotch button buttonhole buy buzz by bye cab cabbage cabin cabinet cackle
cage cake calendar calf call caller calling came camel camp campfire can canal
canary candle candlestick candy cane cannon cannot canoe can't canyon cap cape
capital captain car card cardboard care careful careless carelessness carload
carpenter carpet carriage carrot carry cart carve case cash cashier castle cat
catbird catch catcher caterpillar catfish catsup cattle caught cause cave
ceiling cell cellar cent center cereal certain certainly chain chair chalk
champion chance change chap charge charm chart chase chatter cheap cheat check
checkers cheek cheer cheese cherry chest chew chick chicken chief child
childhood children chill chilly chimney chin china chip chipmunk chocolate
choice choose chop chorus chose chosen christen Christmas church churn
cigarette circle circus citizen city clang clap class classmate classroom claw
clay clean cleaner clear clerk clever click cliff climb clip cloak clock close
closet cloth clothes clothing cloud cloudy clover clown club cluck clump coach
coal coast coat cob cobbler cocoa coconut cocoon cod codfish coffee coffeepot
coin cold collar college color colored colt column comb come comfort comic
coming company compare conductor cone connect coo cook cooked cooking cookie
cookies cool cooler coop copper copy cord cork corn corner correct cost cot
cottage cotton couch cough could couldn't count counter country county course
court cousin cover cow coward cowardly cowboy cozy crab crack cracker cradle
cramps cranberry crank cranky crash crawl crazy cream creamy creek creep crept
cried croak crook crooked crop cross crossing cross-eyed crow crowd crowded
crown cruel crumb crumble crush crust cry cries cub cuff cup cuff cup cupboard
cupful cure curl curly curtain curve cushion custard customer cut cute cutting
dab dad daddy daily dairy daisy dam damage dame damp dance dancer dancing dandy
danger dangerous dare dark darkness darling darn dart dash date daughter dawn
day daybreak daytime dead deaf deal dear death December decide deck deed deep
deer defeat defend defense delight den dentist depend deposit describe desert
deserve desire desk destroy devil dew diamond did didn't die died dies
difference different dig dim dime dine ding-dong dinner dip direct direction
dirt dirty discover dish dislike dismiss ditch dive diver divide do dock doctor
does doesn't dog doll dollar dolly done donkey don't door doorbell doorknob
doorstep dope dot double dough dove down downstairs downtown dozen drag drain
drank draw drawer draw drawing dream dress dresser dressmaker drew dried drift
drill drink drip drive driven driver drop drove drown drowsy drub drum drunk
dry duck due dug dull dumb dump during dust dusty duty dwarf dwell dwelt dying
each eager eagle ear early earn earth east eastern easy eat eaten edge egg eh
eight eighteen eighth eighty either elbow elder eldest electric electricity
elephant eleven elf elm else elsewhere empty end ending enemy engine engineer
English enjoy enough enter envelope equal erase eraser errand escape eve even
evening ever every everybody everyday everyone everything everywhere evil exact
except exchange excited exciting excuse exit expect explain extra eye eyebrow
fable face facing fact factory fail faint fair fairy faith fake fall false
family fan fancy far faraway fare farmer farm farming far-off farther fashion
fast fasten fat father fault favor favorite fear feast feather February fed
feed feel feet fell fellow felt fence fever few fib fiddle field fife fifteen
fifth fifty fig fight figure file fill film finally find fine finger finish
fire firearm firecracker fireplace fireworks firing first fish fisherman fist
fit fits five fix flag flake flame flap flash flashlight flat flea flesh flew
flies flight flip flip-flop float flock flood floor flop flour flow flower
flowery flutter fly foam fog foggy fold folks follow following fond food fool
foolish foot football footprint for forehead forest forget forgive forgot
forgotten fork form fort forth fortune forty forward fought found fountain four
fourteen fourth fox frame free freedom freeze freight French fresh fret Friday
fried friend friendly friendship frighten frog from front frost frown froze
fruit fry fudge fuel full fully fun funny fur furniture further fuzzy gain
gallon gallop game gang garage garbage garden gas gasoline gate gather gave gay
gear geese general gentle gentleman gentlemen geography get getting giant gift
gingerbread girl give given giving glad gladly glance glass glasses gleam glide
glory glove glow glue go going goes goal goat gobble God god godmother gold
golden goldfish golf gone good goods goodbye good-by goodbye good-bye
good-looking goodness goody goose gooseberry got govern government gown grab
gracious grade grain grand grandchild grandchildren granddaughter grandfather
grandma grandmother grandpa grandson grandstand grape grapes grapefruit grass
grasshopper grateful grave gravel graveyard gravy gray graze grease great green
greet grew grind groan grocery ground group grove grow guard guess guest guide
gulf gum gun gunpowder guy ha habit had hadn't hail hair haircut hairpin half
hall halt ham hammer hand handful handkerchief handle handwriting hang happen
happily happiness happy harbor hard hardly hardship hardware hare hark harm
harness harp harvest has hasn't haste hasten hasty hat hatch hatchet hate haul
have haven't having hawk hay hayfield haystack he head headache heal health
healthy heap hear hearing heard heart heat heater heaven heavy he'd heel height
held hell he'll hello helmet help helper helpful hem hen henhouse her hers herd
here here's hero herself he's hey hickory hid hidden hide high highway hill
hillside hilltop hilly him himself hind hint hip hire his hiss history hit
hitch hive ho hoe hog hold holder hole holiday hollow holy home homely homesick
honest honey honeybee honeymoon honk honor hood hoof hook hoop hop hope hopeful
hopeless horn horse horseback horseshoe hose hospital host hot hotel hound hour
house housetop housewife housework how however howl hug huge hum humble hump
hundred hung hunger hungry hunk hunt hunter hurrah hurried hurry hurt husband
hush hut hymn I ice icy I'd idea ideal if ill I'll I'm important impossible
improve in inch inches income indeed Indian indoors ink inn insect inside
instant instead insult intend interested interesting into invite iron is island
isn't it its it's itself I've ivory ivy jacket jacks jail jam January jar jaw
jay jelly jellyfish jerk jig job jockey join joke joking jolly journey joy
joyful joyous judge jug juice juicy July jump June junior junk just keen keep
kept kettle key kick kid kill killed kind kindly kindness king kingdom kiss
kitchen kite kitten kitty knee kneel knew knife knit knives knob knock knot
know known lace lad ladder ladies lady laid lake lamb lame lamp land lane
language lantern lap lard large lash lass last late laugh laundry law lawn
lawyer lay lazy lead leader leaf leak lean leap learn learned least leather
leave leaving led left leg lemon lemonade lend length less lesson let let's
letter letting lettuce level liberty library lice lick lid lie life lift light
lightness lightning like likely liking lily limb lime limp line linen lion lip
list listen lit little live lives lively liver living lizard load loaf loan
loaves lock locomotive log lone lonely lonesome long look lookout loop loose
lord lose loser loss lost lot loud love lovely lover low luck lucky lumber lump
lunch lying ma machine machinery mad made magazine magic maid mail mailbox
mailman major make making male mama mamma man manager mane manger many map
maple marble march March mare mark market marriage married marry mask mast
master mat match matter mattress may May maybe mayor maypole me meadow meal
mean means meant measure meat medicine meet meeting melt member men mend meow
merry mess message met metal mew mice middle midnight might mighty mile milk
milkman mill miler million mind mine miner mint minute mirror mischief miss
Miss misspell mistake misty mitt mitten mix moment Monday money monkey month
moo moon moonlight moose mop more morning morrow moss most mostly mother motor
mount mountain mouse mouth move movie movies moving mow Mr. Mrs. much mud muddy
mug mule multiply murder music must my myself nail name nap napkin narrow nasty
naughty navy near nearby nearly neat neck necktie need needle needn't Negro
neighbor neighborhood neither nerve nest net never nevermore new news newspaper
next nibble nice nickel night nightgown nine nineteen ninety no nobody nod
noise noisy none noon nor north northern nose not note nothing notice November
now nowhere number nurse nut oak oar oatmeal oats obey ocean o'clock October
odd of off offer office officer often oh oil old old-fashioned on once one
onion only onward open or orange orchard order ore organ other otherwise ouch
ought our ours ourselves out outdoors outfit outlaw outline outside outward
oven over overalls overcoat overeat overhead overhear overnight overturn owe
owing owl own owner ox pa pace pack package pad page paid pail pain painful
paint painter painting pair pal palace pale pan pancake pane pansy pants papa
paper parade pardon parent park part partly partner party pass passenger past
paste pasture pat patch path patter pave pavement paw pay payment pea peas
peace peaceful peach peaches peak peanut pear pearl peck peek peel peep peg pen
pencil penny people pepper peppermint perfume perhaps person pet phone piano
pick pickle picnic picture pie piece pig pigeon piggy pile pill pillow pin pine
pineapple pink pint pipe pistol pit pitch pitcher pity place plain plan plane
plant plate platform platter play player playground playhouse playmate
plaything pleasant please pleasure plenty plow plug plum pocket pocketbook poem
point poison poke pole police policeman polish polite pond ponies pony pool
poor pop popcorn popped porch pork possible post postage postman pot potato
potatoes pound pour powder power powerful praise pray prayer prepare present
pretty price prick prince princess print prison prize promise proper protect
proud prove prune public puddle puff pull pump pumpkin punch punish pup pupil
puppy pure purple purse push puss pussy pussycat put putting puzzle quack quart
quarter queen queer question quick quickly quiet quilt quit quite rabbit race
rack radio radish rag rail railroad railway rain rainy rainbow raise raisin
rake ram ran ranch rang rap rapidly rat rate rather rattle raw ray reach read
reader reading ready real really reap rear reason rebuild receive recess record
red redbird redbreast refuse reindeer rejoice remain remember remind remove
rent repair repay repeat report rest return review reward rib ribbon rice rich
rid riddle ride rider riding right rim ring rip ripe rise rising river road
roadside roar roast rob robber robe robin rock rocky rocket rode roll roller
roof room rooster root rope rose rosebud rot rotten rough round route row
rowboat royal rub rubbed rubber rubbish rug rule ruler rumble run rung runner
running rush rust rusty rye sack sad saddle sadness safe safety said sail
sailboat sailor saint salad sale salt same sand sandy sandwich sang sank sap
sash sat satin satisfactory Saturday sausage savage save savings saw say scab
scales scare scarf school schoolboy schoolhouse schoolmaster schoolroom scorch
score scrap scrape scratch scream screen screw scrub sea seal seam search
season seat second secret see seeing seed seek seem seen seesaw select self
selfish sell send sense sent sentence separate September servant serve service
set setting settle settlement seven seventeen seventh seventy several sew shade
shadow shady shake shaker shaking shall shame shan't shape share sharp shave
she she'd she'll she's shear shears shed sheep sheet shelf shell shepherd shine
shining shiny ship shirt shock shoe shoemaker shone shook shoot shop shopping
shore short shot should shoulder shouldn't shout shovel show shower shut shy
sick sickness side sidewalk sideways sigh sight sign silence silent silk sill
silly silver simple sin since sing singer single sink sip sir sis sissy sister
sit sitting six sixteen sixth sixty size skate skater ski skin skip skirt sky
slam slap slate slave sled sleep sleepy sleeve sleigh slept slice slid slide
sling slip slipped slipper slippery slit slow slowly sly smack small smart
smell smile smoke smooth snail snake snap snapping sneeze snow snowy snowball
snowflake snuff snug so soak soap sob socks sod soda sofa soft soil sold
soldier sole some somebody somehow someone something sometime sometimes
somewhere son song soon sore sorrow sorry sort soul sound soup sour south
southern space spade spank sparrow speak speaker spear speech speed spell
spelling spend spent spider spike spill spin spinach spirit spit splash spoil
spoke spook spoon sport spot spread spring springtime sprinkle square squash
squeak squeeze squirrel stable stack stage stair stall stamp stand star stare
start starve state station stay steak steal steam steamboat steamer steel steep
steeple steer stem step stepping stick sticky stiff still stillness sting stir
stitch stock stocking stole stone stood stool stoop stop stopped stopping store
stork stories storm stormy story stove straight strange stranger strap straw
strawberry stream street stretch string strip stripes strong stuck study stuff
stump stung subject such suck sudden suffer sugar suit sum summer sun Sunday
sunflower sung sunk sunlight sunny sunrise sunset sunshine supper suppose sure
surely surface surprise swallow swam swamp swan swat swear sweat sweater sweep
sweet sweetness sweetheart swell swept swift swim swimming swing switch sword
swore table tablecloth tablespoon tablet tack tag tail tailor take taken taking
tale talk talker tall tame tan tank tap tape tar tardy task taste taught tax
tea teach teacher team tear tease teaspoon teeth telephone tell temper ten
tennis tent term terrible test than thank thanks thankful Thanksgiving that
that's the theater thee their them then there these they they'd they'll they're
they've thick thief thimble thin thing think third thirsty thirteen thirty this
thorn those though thought thousand thread three threw throat throne through
throw thrown thumb thunder Thursday thy tick ticket tickle tie tiger tight till
time tin tinkle tiny tip tiptoe tire tired title to toad toadstool toast
tobacco today toe together toilet told tomato tomorrow ton tone tongue tonight
too took tool toot tooth toothbrush toothpick top tore torn toss touch tow
toward towards towel tower town toy trace track trade train tramp trap tray
treasure treat tree trick tricycle tried trim trip trolley trouble truck true
truly trunk trust truth try tub Tuesday tug tulip tumble tune tunnel turkey
turn turtle twelve twenty twice twig twin two ugly umbrella uncle under
understand underwear undress unfair unfinished unfold unfriendly unhappy unhurt
uniform United States unkind unknown unless unpleasant until unwilling up upon
upper upset upside upstairs uptown upward us use used usefulç valentine valley
valuable value vase vegetable velvet very vessel victory view village vine
violet visit visitor voice vote wag wagon waist wait wake waken walk wall
walnut want war warm warn was wash washer washtub wasn't waste watch watchman
water watermelon waterproof wave wax way wayside we weak weakness weaken wealth
weapon wear weary weather weave web we'd wedding Wednesday wee weed week we'll
weep weigh welcome well went were we're west western wet we've whale what
what's wheat wheel when whenever where which while whip whipped whirl whisky
whiskey whisper whistle white who who'd whole who'll whom who's whose why
wicked wide wife wiggle wild wildcat will willing willow win wind windy
windmill window wine wing wink winner winter wipe wire wise wish wit witch with
without woke wolf woman women won wonder wonderful won't wood wooden woodpecker
woods wool woolen word wore work worker workman world worm worn worry worse
worst worth would wouldn't wound wove wrap wrapped wreck wren wring write
writing written wrong wrote wrung yard yarn year yell yellow yes yesterday yet
yolk yonder you you'd you'll young youngster your yours you're yourself
yourselves youth you've
""".lower().split())

# 3000 most frequent word tokens in Sonar 500 corpus
basicwords_nl = frozenset("""
. de , van het een en in dat is op te zijn voor met ik die niet ) ( : " maar er
' aan - ook je als om ? hij ze bij dan nog was naar uit of door we heeft over
wat al tot worden meer hebben wordt geen wel jaar kan ! dit nu zich zo hun deze
werd moet mijn haar na kunnen zou veel tegen ... 1 had ; twee heb zal daar toch
andere goed eerste wil moeten waar mensen / onder nieuwe gaat gaan 2 dus hem u
ons me ben weer hier alleen toen omdat hoe doen zegt onze alle heel tussen
maken mij grote eens '' zij waren uur komt iets komen staat zoals euro wij 3
drie af * volgens tijd want altijd één zelf wie zonder mee weet man vooral
eigen ja tweede enkele leven zelfs plaats keer zien toe dag alles jaren laten
the 4 men echt werden zullen willen nooit weg 0 % zeker kinderen laatste kon
tijdens net week kwam terug mag .. 5 land krijgen europese staan per werk
zeggen binnen minder even procent aantal steeds laat niets blijft 6 miljoen
beter samen ging blijven werken zo'n elkaar zit verder anders rond nederland
ten hadden gewoon lang artikel vier misschien via geven vandaag zei iedereen \]
eerst moest jij re goede waarom 10 sinds pas geleden weten later hele geld
wereld zouden mogelijk hen iemand vraag brussel groot belgië vrouw doet houden
deel bijna elke vinden echter huis gisteren vaak \[ vanaf nodig maakt kreeg
opnieuw vind allemaal zitten snel vlaamse nemen ter volgende vijf krijgt achter
der maanden dagen minister stad erg uw denk natuurlijk da la politie nee derde
belgische nederlandse weinig amerikaanse terwijl ligt ga zeer a helemaal
geweest vorig hebt den 7 duidelijk kleine 20 commissie regering vrouwen daarom
gemaakt wanneer manier jullie beste spelen enige vindt 8 waarin naam soms
eigenlijk verschillende groep zie begin paar tien lijkt bijvoorbeeld eerder
gedaan graag weken onderzoek gezien geval mogen bent 15 europa oude voorzitter
partij vragen geeft genoeg seizoen enkel landen open vorige bovendien frank
meteen vlaanderen wilde moment welke kans zes politieke antwerpen zondag zaken
club stond raad bedrijf mannen elk 12 brengen buiten ooit beetje grootste hand
naast slechts 9 ziet probleem bekend nieuw titel niemand problemen vader maakte
school 25 zaterdag vrij ander markt dood 11 gent thuis morgen dezelfde moeilijk
denken aldus jonge einde daarna le 't bestaat moeder stellen eind vast jan
blijkt boven kijken zetten water boek vanuit muziek 30 zaak kind ouders deed
zowel 14 wedstrijd ` 18 vond bedrijven ploeg zoveel gemeente meeste afgelopen
soort punten meest kunt maand les quote geworden & extra zodat sociale meter
film gebruik gebruikt doe beide kom liggen franse halen vroeg zag gezegd
president politiek dingen uiteindelijk langs = hoop valt houdt september b
sommige miljard lange best d belang des vlaams familie zeg liet zat hoofd | ge
13 bezig à helft nationale neemt juni 16 zorgen ervan sterk jongeren auto et
mei stuk daarmee handen januari niks lid kwamen konden druk trouwens ongeveer
start immers frankrijk 17 internationale nieuws recht vele p belangrijk prijs
2004 oktober bank rol overheid waarop informatie foto burgemeester waarbij ogen
langer vroeger betalen mooi parlement brugge juist echte kant verhaal bepaalde
zichzelf beginnen waarvan volledig verordening inderdaad basis nadat klein hoge
juli zin lopen duitse wist mens zoon economische anderen hetzelfde radio oorlog
werkt toekomst speelt 2005 's gaf horen meestal huidige reden periode stelt
minuten .... acht verwacht totaal gelijk bestaan daarbij 19 vol december succes
god maart 21 daarvoor hoor gehad ruim ni amsterdam begon loopt idee april
begint licht > eten precies kosten bleef slecht hoog duitsland vorm betekent
eeuw financiële vallen rest ver vrijdag zeven woord jou peter kort grond
premier programma bleek wachten genomen 24 spelers 50 22 gegeven terecht 100
gebied ronde verenigde belangrijke zware partijen plaatsen z'n nr. aandacht
november nam waardoor punt rode gelukkig rust dienst nummer plan staten woorden
kijk publiek hoewel rekening dacht gemeenschap situatie zoek unie volgend
voorbije mocht feit zoeken gebeurt intussen new weekend kamer 23 moesten klaar
koning leden doel trekken gebruiken zwaar verkiezingen britse helpen leren + i
belangrijkste vs trainer centrum gekomen gevonden kregen augustus vierde kennen
waarschijnlijk namen dollar mooie volgen dochter «ik waarmee 2003 daarvan
vertelt buitenlandse lijst komende spreken 2000 gevolg tegenover korte proberen
half vrije maandag top won zet gevallen gingen onderwijs straat vlak steun and
winnen daardoor beeld vrienden maatregelen denkt ene bijzonder nederlands
ruimte vormen leggen europees liep geschiedenis finale gevoel inwoners ondanks
volgt niveau 2001 februari praten hard geloof richting strijd leuk buurt cd&v
blz. heer paul gratis antwoord betrokken voorlopig  gekregen 2002 wellicht
verloren volk begonnen directeur gebracht vld lol economie speelde project --
schepen info winst organisatie voetbal werknemers stemmen 40 verschil momenteel
gebeuren brengt wet systeem internet hoger kilometer leuven nacht oud tom alsof
marc viel liever kopen actie woensdag kun betreft ieder sport wonen gehouden
bevolking kerk wegens fc akkoord vraagt straks cijfers lezen orde algemeen
leiden beleid bart normaal uiteraard ma twintig rijden kost hart jongens
producten wagen oog richtlijn l én irak 28 meisje samenwerking blijkbaar kent
hulp erop klanten jongen namelijk macht 26 hoogte nie 27 schrijven graden
genoemd nauwelijks anderlecht verleden boeken waarde heen oplossing 2006
ziekenhuis du eén luc reeks vriend provincie lokale contact controle donderdag
vanavond witte overigens overal avond geboren deur kiezen ondertussen besluit
keuze hield voldoende blij bepaald media sterke zomer zwarte zee geef rechter
liefde lijn bestuur nou team taal beroep tekst resultaat regio to wou
resultaten oostende dinsdag krant termijn voorbeeld indien slag vertrouwen
doelpunten stand kennis s voorstel inmiddels hoeveel wijze 31 geweld vergeten
david voorbij vervolgens dieren mening gezet groter dankzij woning frans voeren
brand foto's bezoek «de m gesteld telkens 60 hé vertellen leeftijd ontwikkeling
ervaring cultuur geldt lag plannen krijg meisjes aandeel gevraagd km kunst
liefst nogal ervoor china leger lidstaten leerlingen haalde kracht bracht
sprake dicht johan dergelijke ne aandelen crisis los stap groen meerderheid
bezoekers kritiek gewonnen parijs actief eindelijk bedoeling halve persoon ken
tevreden legt iedere delen beslissing 29 relatie kop personen gezicht schreef
betere leek israël leider behalve noemen fout mechelen slechte jong geheel
wacht john daarnaast verkocht gevolgen daarin trekt bang negen bieden c ergens
universiteit klasse voorkomen gang jongste maak gegevens lichaam vervangen
lekker opdracht ii stelde omgeving gebeurd neem minstens direct 2007 «we
slachtoffer groei rug lucht brusselse prijzen bekende verkeer personeel raakte
he gegaan gebouw patrick name kaart gij hou onlangs gewone pakken kwaliteit
twaalf wilt heet scholen moeite midden links doden juiste geschreven «het
verkoop kader amerika tafel voelen stem website bal york voorzien regels sector
museum gesloten advies slachtoffers hoeft biedt contract kansen verkopen jouw
openbare k last plus italiaanse band spel wedstrijden rechtbank verband alvast
indruk hoort activiteiten genk federale 80 gezin antwerpse hoogste un italië
bouwen productie politici voorstellen proces mogelijke beslist spanje _ neer
invloed dertig middelen verslag spreekt gespeeld dirk word opgenomen pb
vastgesteld gebeurde omzet geloven zette 70 leiding zoiets klinkt justitie
stonden telt on amper wilden inzake discussie zon for optreden algemene
gemeenten o feiten peeters ministerie veld dans e moord ontstaan gevaar stoppen
speciale match duur dragen belgen veiligheid aanwezig centrale beperkt trok t
daarop 35 tegenwoordig onmiddellijk loop bewoners rusland hoofdstad verliezen
tv veranderen verlies baan keren kwestie buitenland honderd schade vuur rustig
tel. bed londen 00 brief café eenmaal fr. broer diverse jeugd kortrijk voel
bush vaste gesprek bijlage 1999 merk openbaar mevrouw enorm bron maatschappij
samenleving beelden wim werkte spaanse dank vanwege bedrag sp.a voelt reactie
gehoord dient steden ok beschikking pijn aarde wa toepassing stel schrijft 200
gekozen collega's burgers allerlei dorp hangt post erin provinciale positie cel
internationaal eisen geel haalt 90 advocaat koers 1996 enorme persoonlijke
verlaten kwijt zolang absoluut banken beurs gebrek vriendin goeie kabinet
stappen zege nochtans energie verloor sluiten functie binnenkort positief
standard vrijheid afstand plek winkel welk zorgt voormalige volle verdrag
leterme sociaal hans tegelijk studenten dl rotterdam geluk zowat jammer
roeselare mogelijkheden amerikanen moderne tonen bestaande reed noemt televisie
makkelijk debat mis lijken haag gemiddelde carrière site officiële aanleiding
veroordeeld diensten 1995 bereiken reeds eu sneller regelmatig plots eveneens
geraakt westen wk zorg hof technische das leidt grotere initiatief militaire
auto's betaald terrein hopen russische vijftien grens ontvangen woont pour
sommigen rechten omstandigheden bureau voelde groepen vakantie bereid groene
gewerkt val sfeer respect vlamingen bewijzen oh mogelijkheid gouden probeert
verwachten brandweer rij volledige gemiddeld vijfde hotel rekenen bestuurder
overwegende bedoeld geert wint regen perfect verantwoordelijk persoonlijk blijf
voordeel vijftig product moslims dienen wind n bus comité totale opleiding
redenen gesproken une kijkt risico hasselt voet deden dagelijks waarvoor
beweging nadien kwaad dak bereikt voorsprong vonden gericht klassieke officieel
turkije michel gemakkelijk \\ zaal hangen zaten 1998 werking drinken
overwinning michael films mezelf kaarten handel stil rijk noch droog zomaar
voorwaarden meneer schuld se verhalen vakbonden natuur aangezien verplicht
overeenkomst bedraagt steken rapport mond nodige maakten helaas prins zuiden
dossier uren diep erbij eerlijk studie as pak eigenaar slaan hogere klopt wegen
genieten bergen tour charleroi betreffende bloed uitgevoerd
verantwoordelijkheid jaarlijks thomas kampioen overleden sprak 2008 rechts
vaker leveren coach sturen job overleg station keek you allen parket waaronder
centraal deelnemers gestolen pieter reis nederlanders uitspraak alweer waard
boodschap zwart verklaring engeland speler duizenden toegang tijden eg publieke
gedacht opgepakt h f verboden geregeld bewust lacht angst eraan geplaatst sorry
park que elf bouw zulke cd begrijpen gek ware gele bedankt vereniging voordat
probeerde schrijver gevangenis islam koop hoopt positieve huizen laag waregem
competitie kleur guy wijn versie slapen verdwenen karakter honderden overtuigd
ah harde speciaal onmogelijk bos praktijk 45 albert zoekt waarheid zagen
belgisch zogenaamde di gehaald tientallen noorden 1994 gisteravond koen
afdeling boete lager gewond hoorde chinese rood boom aanpak 75 daarover dokter
nergens beneden directie gevolgd woningen 32 ongeval wees tuin gedurende ding
merken brussels stukken gebouwd rechtstreeks doelman degelijk gentse zekerheid
gedrag democratie toont vooruit 500 x veilig huwelijk passen mijnheer
tentoonstelling aalst lachen wakker wijzen warm baas soorten hond leeft m'n
gewijzigd daders karel vóór ermee onderzoeken qui bert wallonië bezit campagne
klant verdwijnen keuken doordat velen tijdje partner steven welkom ziekte
bijzondere principe raken robert sk effect lichte concurrentie besloten
reageren grenzen groetjes neen medewerkers materiaal ontslag engels turkse
levert model gebieden ofwel lage afgesloten poging ouder hoek serieus
historische voortaan bekijken vreemd vergelijking volgde kern starten veranderd
onderhandelingen herman draait probeer muur jezelf restaurant agenten ogenblik
noodzakelijk fiets gebouwen stijl au vrede aanvankelijk ziek taak vlucht kust
aanbod kim verkeerd engelse duizend verdienen beurt vroegere vertelde 33 eiland
kevin wijst sta begrip past roman onderwerp fortis kandidaten feest flink legde
mate collega soldaten betrekking voeten ondernemingen opgericht moeilijke
oorzaak wijk vlees hè 'n bescherming geest straf woordvoerder japan trein
meerdere verschillen tenminste est traditionele draaien behandeling 1997 zover
combinatie organiseren smaak ie beiden wereldoorlog computer milieu forum vormt
wél gelegen vertrekken recente type limburg oudere wit lief plezier voortdurend
ton manager zestig voort spoor lieten eur verbonden polen avonds bef projecten
tevens j. zorgde inhoud elders reacties il gemeenteraad kende tachtig janssens
uitgebreid vrijwel winnaar bewijs lees vzw veertig neus middel opvallend banen
eeg baby iran instellingen gelegenheid onderneming leuke gerecht vanmorgen
dikke generatie yves festival begrijp werkelijkheid toestand afrika bedoelde
luik nationaal betaalt selectie financieel vielen el eric winter oudste
economisch zelden afkomstig besloot evenwel commerciële reageert bevat c8 blok
sluit franstalige spijt thuisploeg bedoel aanval slot ploegen kv duren mark
dieven leidde vertrek ministers vandaar dichter aanslag organiseert verdere
spits begroting sterker luisteren ideeën verhofstadt podium klacht veroorzaakt
maria waalse plaatselijke acties n-va doelpunt israëlische willem blijken
behoorlijk beschikbaar georganiseerd telefoon mochten wapens par gebaseerd live
twijfel 65 eruit middag bleven 150 olympische operatie 300 oppervlakte
organisaties tegenstelling it anderzijds duurt goe beschouwd drugs deuren pers
oppositie afscheid binnenlandse thema 34 jezus definitief bepalingen serie
louis snelheid relatief bepalen verkeerde blik ongetwijfeld winkels culturele
papier joden afhankelijk olie minuut snelle kandidaat utrecht stof eer
weliswaar eventueel ernstig 36 interessant oosten mama lot 99 hoeven groeien
achteraf gelezen international bod plaatse afghanistan vergadering redactie
prima vervoer procedure bellen helpt allez zo. b. 1992 praat 1993 sloeg belg
zodra redden m. oké min onderdeel uitleg netwerk scoorde lukt willy ontdekt
hoofdpunten trots tenslotte lokeren groot-brittannië uitbreiding droom p. arm
fase clubs erger training filip komst tc gesprekken 46 eentje specifieke
medische oprichting set behoort aanslagen houding armen missen topic studio
betekenis jaarlijkse vliegen behouden mooiste getroffen berichten waarden
stijging militairen programma's bijdrage stijn financiën vb visie
belangstelling glas hopelijk seconden verzoek gerust sindsdien koningin vis
eenvoudig haal vermijden andré meent standaard gelegd dreigt talent destijds
bomen koffie da's gekocht ps draagt bestond 1991 gevaarlijk philippe d.
commentaar roepen 1990 patiënten vliegtuig verjaardag gasten aanwezigheid zak
chris <p> fouten ineens fijn gezondheid theater zult 02 inzet instituut
palestijnse teken werkelijk brug wisten gezond troepen uitvoering luchthaven
gemeenschappen auteur reizen dames rock george verzet vrees democratische
kranten gezinnen bericht kleuren vermoord koninklijke defensie benen menselijke
vreemde 55 turnhout joodse allebei redelijk a. editie kindje haast kiest stukje
pierre eddy jos gebleven partners jarenlang investeringen zus gelet datum
standpunt vermoedelijk waarna < motor sur voorstelling vergeet maxima
commissaris professor beschikken zeventig japanse minste martin erik vooraf
literatuur voorkeur kwart euh schip technisch roept voorgesteld show scoren
india sporen slagen tekort prachtige individuele 01 regionale gezocht inkomsten
voornamelijk 38 compleet negatieve nederlaag sven lagere beveren boonen wens
vermogen v. conflict goedgekeurd dubbele geslaagd 98 teksten hallo stichting
toevallig goud negentig pagina tegenstander antwoorden koninkrijk 37 traditie
roger mist behoren stijgen stadsbestuur letterlijk mekaar kleinere vrt westerse
natuurlijke interview straten toon gsm kris goederen zekere geleid belastingen
g beleggers vanmiddag haven rijdt scheidsrechter overeenkomstig streek speelden
wetenschap hiervoor klachten industrie zone uitzondering richten arbeiders
pakte zesde burger slaat inkomen ervaren ontwikkelen prachtig tegelijkertijd
eventuele verklaarde washington tijdelijk link rand zingen bel seks kleren
hiermee communautaire republiek agenda hugo onafhankelijk voert verdeeld wagens
digitale momenten leiders gemeentebestuur gedachten externe heilige vannacht
rome aangeboden gewezen lierse regeling waaraan an 42 hemel geheim islamitische
planten ontslagen noemde tja beschermen aangepast gestuurd instelling raakt
verscheen verdient concert oranje vrouwelijke spa kwartaal absolute feite stop
ruzie verschenen fans verschijnen buren 250 onvoldoende geschorst fietsen
duidelijke 97 voldoen idd qua interne jean ambtenaren been steunen wereldwijd
supporters beslissen gewest ek verbeteren verdachte renners steeg tienen
voorbereiding verandering zicht treden dromen madrid reclame interesse nood
dader sterven oostenrijk aanzien world conclusie normale omhoog sloot
lid-staten behoefte ontstond papa leverde management consument blauwe
uitsluitend omroep heren rit ingediend durven landbouw overname langzaam beker
simpel paus diens uiterlijk verteld bier evolutie begrepen brede verliest
walter kantoor beschikt nummers test kamp my league geleerd 400 jacques
afspraken maximaal gaven pensioen koos dringend aangehouden autoriteiten tal
ernstige achtergrond dr. luister wetgeving dure dier volop 44 allochtonen
hoeveelheid instantie es opgesteld daarentegen piet tenzij geluid behandeld
bewezen menen contacten steve verdediging favoriete y geraken maat boeren
ongeluk gelooft uitvoeren geopend stoffen afloop miljoenen schoenen c. riep
anderhalf moeilijker journalisten beslag lieve adres fonds maatschappelijke
gedeelte analyse meester klas tony wetenschappelijke regisseur boot katholieke
c6 markten dendermonde gemeenschappelijke gast vn mede san bezoeken arts out
gebruikte belangen afspraak nachts alsnog liepen investeren gelden categorie
hierbij getuigen elementen blad vlot bond ramp treedt trap ring normen teveel
prestatie tim bloemen verandert mol grotendeels voedsel bijdragen 85 opvolger
overige gevoelens gebeurtenissen zwanger uitspraken kunstenaar zulte eindigde
logisch aard verdedigen omwille berlijn communicatie waarover usd moe gestegen
ontwikkeld wensen g. strategie alternatief vogels vaststelling 39 stilaan
chauffeur geprobeerd zetels 43 slaap evenwicht raakten gezelschap geweldig
album dubbel lagen wezen huid erover kunstenaars voormalig leest opnemen banden
cercle fiscale aanleg dik deelnemen wolken verdacht united voertuig rijke u.
onbekende 1000 record kristof onderzocht opklaringen warme structuur ideale
ochtend veertien 48 bekeken stuur voorwaarde integratie sneeuw groeit tellen
openen vertrokken oorsprong aanmerking creëren studies be v pvda km² ajax gat
populair nederlander ingezet 1989 portugal belasting negatief uiterst grondwet
beheer moskou vloer roken artsen bevindt stevig prestaties lijden onderzoekers
stadion gedragen evenmin schuldig brood ach bijkomende formule tekenen verre
luidt iii voren budget s. voorbeelden overzicht hierdoor lastig techniek vos
vrachtwagen oudenaarde technologie duitsers enig gaten gelijke zwitserland
koppel berg ce trouw loon collectie werkgevers koud leeg eh guido opening
beperkte voorzichtig vermeld gepland fabriek rente richard toekomstige schoot
eet stopt bob aandeelhouders schrik controleren christelijke toeristen city
risico's 20.00 james kilo gevoerd breed geschikt vluchten oorspronkelijke
sint-niklaas kapitaal kanten greep werkloosheid maal oren minst 41 schorsing
rolleyes belangrijker smet inspanningen historisch pakt real getrokken
kruispunt vorst that zweden trek love st. las beperken verhouding verwachtingen
bestuurders liter toonde evenveel opzichte firma danny arabische verklaard
breken france westerlo achterstand hoi leert raar uitslag € afrikaanse vandaan
journalist klagen geslagen charles kwartier maes zakken schoon onafhankelijke
liedje gewicht realiteit socialistische werkgever meldt zen stevige stemming
bedragen terugkeer aa barcelona boos regel details kleiner tmf klap tis acteur
passagiers aardig spanning gedachte big verscheidene vechten finales schat
fransen concrete rik oproep goedkoper verenigd onzer slotte protest jo bruno
uitstekend integendeel schieten imago onderdelen theorie gedwongen franstaligen
locatie 20.30 overleed cda tegenstanders zevende zwakke concept martens
overtuigen bronnen constant populaire boord jeroen pink fantastisch ingang
linkse college amsterdamse belga religieuze opent onafhankelijkheid vieren
toerisme no haat politicus 2009 congo luxemburg meegemaakt iraakse golf aankoop
gooien kasteel tips coalitie mss betekenen revolutie overeenstemming redacteur
bibliotheek saddam verklaart maatregel champions marcel kiezers dode «in 52
brak vtm zaventem zijde overgenomen reageerde arme super hetgeen paard naartoe
juridische naties zestien opmerkelijk dagelijkse vroege geweten melk vvd stilte
hielden identiteit aangenomen 2010 getrouwd dertien leer pleit portefeuille dom
respectievelijk duo diezelfde makkelijker stak rob broers strand vergelijken
daartoe griekse staatssecretaris lijf kosovo cheer broek hout stijgt toezicht
dossiers fors voordelen e-mail zichtbaar regime @ sowieso jury papieren koude
melden ontwerp verkozen h. toeval dikwijls doorgaans namens verrassing
bevestigd marokkaanse applaus beloften cup geheime vluchtelingen toegepast
leeuw verschijnt schaal australië geïnteresseerd effectief danken armoede von
scherp gewonden wetten voorts automatisch massaal vrezen bord stroom houten
jongere zanger inclusief alcohol uitgesproken ruime cm geconfronteerd stelling
fusie geslacht verlopen pakistan plant zuid-afrika vooruitgang verhogen knie
aparte gepleegd
""".split())

# 1000 MFW German; http://www.wortschatz.uni-leipzig.de/Papers/top1000de.txt
basicwords_de = frozenset("""
der die und in den von zu das mit sich des auf für ist im dem nicht ein Die
eine als auch es an werden aus er hat daß sie nach wird bei einer Der um am
sind noch wie einem über einen Das so Sie zum war haben nur oder aber vor zur
bis mehr durch man sein wurde sei In Prozent hatte kann gegen vom können schon
wenn habe seine Mark ihre dann unter wir soll ich eines Es Jahr zwei Jahren
diese dieser wieder keine Uhr seiner worden Und will zwischen Im immer
Millionen Ein was sagte Er gibt alle DM diesem seit muß wurden beim doch jetzt
waren drei Jahre Mit neue neuen damit bereits da Auch ihr seinen müssen ab
ihrer Nach ohne sondern selbst ersten nun etwa Bei heute ihren weil ihm seien
Menschen Deutschland anderen werde Ich sagt Wir Eine rund Für Aber ihn Ende
jedoch Zeit sollen ins Wenn So seinem uns Stadt geht Doch sehr hier ganz erst
wollen Berlin vor allem sowie hatten kein deutschen machen lassen Als
Unternehmen andere ob dieses steht dabei wegen weiter denn beiden einmal etwas
Wie nichts allerdings vier gut viele wo viel dort alles Auf wäre SPD kommt
vergangenen denen fast fünf könnte nicht nur hätten Frau Am dafür kommen diesen
letzten zwar Diese großen dazu Von Mann Da sollte würde also bisher Leben
Milliarden Welt Regierung konnte ihrem Frauen während Land zehn würden stehen
ja USA heißt dies zurück Kinder dessen ihnen deren sogar Frage gewesen erste
gab liegt gar davon gestern geben Teil Polizei dass hätte eigenen kaum sieht
große Denn weitere Was sehen macht Angaben weniger gerade läßt Geld München
deutsche allen darauf wohl später könne deshalb aller kam Arbeit mich gegenüber
nächsten bleibt wenig lange gemacht Wer Dies Fall mir gehen Berliner mal Weg
CDU wollte sechs keinen Woche dagegen alten möglich gilt erklärte müsse Dabei
könnten Geschichte zusammen finden Tag Art erhalten Man Dollar Wochen jeder nie
bleiben besonders Jahres Deutschen Den Zu zunächst derzeit allein deutlich
Entwicklung weiß einige sollten Präsident geworden statt Bonn Platz inzwischen
Nur Freitag Um pro seines Damit Montag Europa schließlich Sonntag einfach
gehört eher oft Zahl neben hält weit Partei meisten Thema zeigt Politik Aus
zweiten Januar insgesamt je mußte Anfang hinter ebenfalls ging Mitarbeiter
darüber vielen Ziel darf Seite fest hin erklärt Namen Haus An Frankfurt
Gesellschaft Mittwoch damals Dienstag Hilfe Mai Markt Seit Tage Donnerstag
halten gleich nehmen solche Entscheidung besser alte Leute Ergebnis Samstag Daß
sagen System März tun Monaten kleinen lang Nicht knapp bringen wissen Kosten
Erfolg bekannt findet daran künftig wer acht Grünen schnell Grund scheint
Zukunft Stuttgart bin liegen politischen Gruppe Rolle stellt Juni sieben
September nämlich Männer Oktober Mrd überhaupt eigene Dann gegeben Außerdem
Stunden eigentlich Meter ließ Probleme vielleicht ebenso Bereich zum Beispiel
Bis Höhe Familie Während Bild Ländern Informationen Frankreich Tagen schwer
zuvor Vor genau April stellen neu erwartet Hamburg sicher führen Mal Über
mehrere Wirtschaft Mio Programm offenbar Hier weiteren natürlich konnten stark
Dezember Juli ganze kommenden Kunden bekommen eben kleine trotz wirklich Lage
Länder leicht gekommen Spiel laut November kurz politische führt innerhalb
unsere meint immer wieder Form Münchner AG anders ihres völlig beispielsweise
gute bislang August Hand jede GmbH Film Minuten erreicht beide Musik Kritik
Mitte Verfügung Buch dürfen Unter jeweils einigen Zum Umsatz spielen Daten
welche müßten hieß paar nachdem Kunst Euro gebracht Problem Noch jeden Ihre
Sprecher recht erneut längst europäischen Sein Eltern Beginn besteht Seine
mindestens machte Jetzt bietet außerdem Bürger Trainer bald Deutsche Schon
Fragen klar Durch Seiten gehören Dort erstmals Februar zeigen Titel Stück
größten FDP setzt Wert Frankfurter Staat möchte daher wolle Bundesregierung
lediglich Nacht Krieg Opfer Tod nimmt Firma zuletzt Werk hohen leben unter
anderem Dieser Kirche weiterhin gebe gestellt Mitglieder Rahmen zweite Paris
Situation gefunden Wochenende internationalen Wasser Recht sonst stand Hälfte
Möglichkeit versucht blieb junge Mehrheit Straße Sache arbeiten Monate Mutter
berichtet letzte Gericht wollten Ihr zwölf zumindest Wahl genug Weise Vater
Bericht amerikanischen hoch beginnt Wort obwohl Kopf spielt Interesse Westen
verloren Preis Erst jedem erreichen setzen spricht früher teilte Landes zudem
einzelnen bereit Blick Druck Bayern Kilometer gemeinsam Bedeutung Chance
Politiker Dazu Zwei besten Ansicht endlich Stelle direkt Beim Bevölkerung Viele
solchen Alle solle jungen Einsatz richtig größte sofort neuer ehemaligen
unserer dürfte schaffen Augen Rußland Internet Allerdings Raum Mannschaft neun
kamen Ausstellung Zeiten Dem einzige meine Nun Verfahren Angebot Richtung
Projekt niemand Kampf weder tatsächlich Personen dpa Heute geführt Gespräch
Kreis Hamburger Schule guten Hauptstadt durchaus Zusammenarbeit darin Amt
Schritt meist groß zufolge Sprache Region Punkte Vergleich genommen gleichen du
Ob Soldaten Universität verschiedenen Kollegen neues Bürgermeister Angst
stellte Sommer danach anderer gesagt Sicherheit Macht Bau handelt Folge Bilder
lag Osten Handel sprach Aufgabe Chef frei dennoch DDR hohe Firmen bzw Koalition
Mädchen Zur entwickelt fand Diskussion bringt Deshalb Hause Gefahr per zugleich
früheren dadurch ganzen abend erzählt Streit Vergangenheit Parteien
Verhandlungen jedenfalls gesehen französischen Trotz darunter Spieler forderte
Beispiel Meinung wenigen Publikum sowohl meinte mag Auto Lösung Boden Einen
Präsidenten hinaus Zwar verletzt weltweit Sohn bevor Peter mußten keiner
Produktion Ort braucht Zusammenhang Kind Verein sprechen Aktien gleichzeitig
London sogenannten Richter geplant Italien Mittel her freilich Mensch großer
Bonner wenige öffentlichen Unterstützung dritten nahm Bundesrepublik
Arbeitsplätze bedeutet Feld Dr. Bank oben gesetzt Ausland Ministerpräsident
Vertreter z.B. jedes ziehen Parlament berichtete Dieses China aufgrund Stellen
warum Kindern heraus heutigen Anteil Herr Öffentlichkeit Abend Selbst Liebe
Neben rechnen fällt New York Industrie WELT Stuttgarter wären Vorjahr Sicht
Idee Banken verlassen Leiter Bühne insbesondere offen stets Theater ändern
entschieden Staaten Experten Gesetz Geschäft Tochter angesichts gelten Mehr
erwarten läuft fordert Japan Sieg Ist Stimmen wählen russischen gewinnen CSU
bieten Nähe jährlich Bremen Schüler Rede Funktion Zuschauer hingegen anderes
Führung Besucher Drittel Moskau immerhin Vorsitzende Urteil Schließlich Kultur
betonte mittlerweile Saison Konzept suchen Zahlen Roman Gewalt Köln gesamte
indem EU Stunde ehemalige Auftrag entscheiden genannt tragen Börse langen
häufig Chancen Vor allem Position alt Luft Studenten übernehmen stärker ohnehin
zeigte geplanten Reihe darum verhindern begann Medien verkauft Minister wichtig
amerikanische sah gesamten einst verwendet vorbei Behörden helfen Folgen
bezeichnet
""".lower().split())

# 3558 MFW French; CATACH, N. (1985). Les listes orthographiques de base du français. Nathan, Paris
basicwords_fr = frozenset("""
A À ABANDONNER ABBÉ ABORD ABSENCE ABSOLU ABSOLUMENT ACCENT ACCEPTER ACCIDENT ACCIDENTS 
ACCOMPAGNAIENT ACCOMPAGNAIT ACCOMPAGNER ACCOMPLIR ACCOMPLIT ACCORD ACCORDER ACCORDS 
ACHETER ACHEVER ACTE ACTES ACTION ACTIONS ACTIVITÉ ACTIVITÉS ACTUEL ACTUELLE ACTUELLEMENT 
ACTUELLES ACTUELS ADIEU ADMET ADMETTRE ADMINISTRATION ADMINISTRATIONS ADMIRABLE ADMIRABLES 
ADMIRER ADMIS ADRESSER AFFAIRE AFFAIRES AFFECTION AFFIRMER AFFREUSES AFFREUX AFIN AFIN DE 
ÂGE AGENT AGENTS ÂGES AGIR AGISSAIT AGISSANT AGIT AGIT-IL S'AGIT-IL AGRÉABLE AH AI AIDER 
AIE AIENT AIES AILE AILES AILLEURS AIMABLE AIMABLES AIMAIS AIMAIT AIME AIMÉ AIMÉE AIMÉES 
AIMENT AIMER AIMERAIS AIMES AIMĖS AIMEZ AIMIEZ AIMIONS AIMONS AINSI AIR AIRS AIT AJOUTANT 
AJOUTENT AJOUTER AJOUTONS ALLAIENT ALLAIS ALLAIT ALLANT ALLÉE ALLEMAND ALLEMANDE ALLEMANDES 
ALLEMANDS ALLER ALLÉS ALLEZ ALLIEZ ALLIONS ALLONS ALORS AMANT AMANTS ÂME AMÈNENT AMENER 
AMÉRICAIN AMÉRICAINES AMÉRICAINS ÂMES AMI AMIS AMITIÉ AMITIÉS AMOUR AMOURS AMUSER AN ANCIEN 
ANCINNES ANCIENS ANGE ANGES ANGLAIS ANGLAISES ANGOISSE ANGOISSES ANIMAL ANIMAUX ANNÉE ANNÉES 
ANNONÇAIT ANNONCER ANS AOÛT APERCEVAIT APERCEVOIR APERÇOIT APERÇÛT APPARAISSAIT APPARAISSENT 
APPARAÎT APPARAÎTRE APPAREIL APPAREILS APPARENCE APPARENCES APPARTEMENT APPARTEMENTS 
APPARTENAIT APPARTENANT APPARTENIR APPARTIENNENT APPARTIENT APPARUT APPEL APPELAIT APPELÉE 
APPELER APPELEZ APPELLE APPELLENT APPELS APPORTAIT APPORTENT APPORTER APPRENAIT APPREND 
APPRENDRE APPRIS APPRISE APPRISES APPRIT APPROCHAIT APPROCHER APRÈS APRÈS-MIDI ARBRE ARBRES 
ARGENT ARME ARMÉE ARMÉES ARMES ARRACHER ARRANGER ARRÊTÉ ARRÊTÉE ARRÊTÉES ARRÊTENT ARRÊTER 
ARRÊTÉS ARRIÈRE ARRIVAIENT ARRIVAIT ARRIVE ARRIVÉ ARRIVÉE ARRIVÉES ARRIVENT ARRIVER ARRIVÉS 
ARRIVONS ART ARTICLE ARTICLES ARTISTE ARTISTES ARTS ARTS ET MÉTIERS AS ASPECT ASPECTS ASSEOIR 
ASSEYEZ ASSEZ ASSIS ASSISE ASSISES ASSISTER ASSURAIT ASSURER ATMOSPHÈRE ATTACHER ATTEIGNENT 
ATTEIGNIT ATTEINDRE ATTEINT ATTEND ATTENDAIENT ATTENDAIS ATTENDAIT ATTENDANT ATTENDENT 
ATTENDEZ ATTENDIEZ ATTENDONS ATTENDRE ATTENDS ATTENTION ATTIRENT ATTIRER ATTITUDE ATTITUDES 
AU AU-DESSUS AUCUN AUCUNE AUJOURD'HUI AUPRÈS AURA AURAIENT AURAIS AURAIT AURAS AUREZ AURIEZ 
AURIONS AURONS AURONT AUSSI AUSSITÔT AUTANT AUTEUR AUTEURS AUTOMNE AUTOMOBILE AUTOMOBILES 
AUTORITÉ AUTORITÉS AUTOUR AUTRE AUTREFOIS AUTREMENT AUTRES AUX AVAIENT AVAIS AVAIT AVANCE 
AVANCER AVANT AVANTAGE AVANTAGES AVEC AVENIR AVENTURE AVENTURES AVENUE AVEZ AVIEZ AVION AVIONS 
AVIS AVOIR AVONS AVOUE AVOUER AVOUEZ AVRIL AYANT AYEZ AYONS BAIN BAINS BAISER BANQUE BANQUES 
BAS BASE BASES BASSE BASSES BAT BATAILLE BATEAU BATEAUX BÂTIMENT BATTAIT BATTANT BATTRE BATTUS 
BEAU BEAUCOUP BEAUTÉ BEAUTÉS BEAUX BELLE BELLES BESOIN BESOINS BÊTE BÊTES BIEN BIENS BIENTÔT 
BILLET BILLETS BLANC BLANCHE BLANCHES BLANCS BLEU BLEUE BLEUES BLEUS BOIRE BOIS BOIT BOÎTE 
BOÎTES BON BONHEUR BONHOMME BONJOUR BONNE BONNES BONS BORD BORDS BOUCHE BOUGER BOURGEOIS BOUT 
BOUT À BOUT BOUT EN BOUT BOUTEILLE BOUTEILLES BRANCHE BRANCHES BRAS BRAS ET JAMBES BRAVE BREF 
BRILLANT BRILLANTE BRILLANTES BRILLANTS BRUIT BRUITS BRÛLER BRUSQUEMENT BUREAU BUREAUX BUT BUTS 
BUVAIT C' ÇA CABINET CACHAIT CACHÉ CACHÉE CACHÉES CACHENT CACHER CACHÉS CAFÉ CALME CALMES 
CAMARADE CAMARADES CAMION CAMP CAMPAGNE CAMPAGNES CANARD CAPABLE CAPABLES CAPITAINE CAR 
CARACTÈRE CARACTÈRES CARTE CARTES CAS AU CAS OÙ CASSER CAUSAIT CAUSE À CAUSE DE CAUSER 
CAUSES CE CECI CÉDER CELA CÉLÈBRE CÉLÈBRES CELLE CELLE-LÀ CENT CENTRE CENTRES CENTS CEPENDANT 
CERTAIN CERTAINE CERTAINEMENT CERTAINES CERTAINS CERTES CES CESSAIT CESSE CESSENT CESSER CET 
CETTE CEUX CHACUN CHACUNE CHAIR CHAISE CHAISES CHALEUR CHAMBRE CHAMBRES CHAMP CHAMPS CHANCE 
CHANCES CHANGÉ CHANGÉE CHANGÉES CHANGEMENT CHANGER CHANGÉS CHANT CHANTAIENT CHANTAIT CHANTANT 
CHANTENT CHANTER CHANTS CHAPEAU CHAPITRE CHAPITRES CHAQUE CHARGÉ CHARGÉE CHARGÉES CHARGER 
CHARGÉS CHARMANT CHARMANTES CHARMANTS CHARME CHARMES CHASSE CHÀTEAU CHAUD CHAUDES CHAUDS 
CHAUFFAGE CHAUFFER CHEF CHEFS CHEMIN CHER CHERCHAIS CHERCHAIT CHERCHANT CHERCHE CHERCHENT 
CHERCHER CHERCHEZ CHERCHONS CHÈRE CHERS CHEVAL CHEVAUX CHEVEU CHEVEUX CHEZ CHIEN CHIENS 
CHOISIR CHOSE CHOSES CHRÉTIEN CHRÉTIENS CI -CI CIEL CINEMA CINQ CINQUANTE CIRCONSTANCE 
CIRCONSTANCES CITÉ CITER CIVILISATION CIVILISATIONS CLAIR CLAIRES CLAIRS CLASSE CLASSES CLEF 
CLEFS COEUR COEURS COIN COINS COLÈRE COLLER COLONIE COLONIES COLONNE COLONNES COMBAT COMBATS 
COMBIEN COMME COMMENÇAIENT COMMENÇAIT COMMENCE COMMENCÉ COMMENCÉE COMMENCENT COMMENCER COMMENÇONS 
COMMENT COMMERCE COMMETTRE COMMIS COMMUN COMMUNE COMMUNES COMMUNS COMPAGNIE COMPAGNIES COMPLET 
COMPLÈTE COMPLÈTEMENT COMPLETS COMPORTAIT COMPORTANT COMPORTENT COMPORTER COMPOSENT COMPOSER 
COMPRENAIT COMPREND COMPRENDRAIT COMPRENDRE COMPRENDS COMPRENEZ COMPRENNENT COMPRIS COMPRIT 
COMPTAIENT COMPTAIT COMPTE COMPTER COMPTES COMTE CONCEPTION CONCEPTIONS CONCERNAIT CONCERNANT 
CONCERNE CONCERNENT CONCERNER CONCEVOIR CONCLURE CONCLUT CONÇOIT CONCOURS CONDITION CONDITIONS 
CONDUIRE CONDUISAIT CONDUIT CONFIANCE CONFIE CONFIER CONNAIS CONNAISSAIS CONNAISSAIT CONNAISSANCE 
CONNAISSANCES CONNAISSENT CONNAISSEZ CONNAISSONS CONNAÎT CONNAÎTRE CONNU CONNUE CONNUES CONNUS 
CONNUT CONSCIENCE CONSEIL CONSEILS CONSÉQUENCE CONSÉQUENCES CONSERVAIT CONSERVER CONSIDÉRABLE 
CONSIDÉRABLES CONSIDÉRAIT CONSIDÉRANT CONSIDÉRÉE CONSIDÉRER CONSIDÉRONS CONSTATER CONSTITUE 
CONSTITUENT CONSTITUER CONSTRUIRE CONTACT CONTACTS CONTENAIT CONTENANT CONTENIR CONTENT 
CONTENTER CONTENUE CONTIENNENT CONTIENT CONTINUAIT CONTINUE CONTINUENT CONTINUER CONTRAIRE 
CONTRE CONVENIR CONVERSATION CONVERSATIONS CONVIENT CORPS CORPS ET ÂME CORPS ET BIENS CORRESPOND 
CORRESPONDANT CORRESPONDENT CORRESPONDRE CÔTE CÔTÉ CÔTES CÔTÉS COU COUCHÉ COUCHÉE COUCHÉES COUCHER 
COUCHÉS COULER COULEUR COUP COUPAIT COUPER COUPS COUR COURAGE COURAIT COURANT COURANTS COURIR COURS 
COURSE COURSES COURT COURTES COURTS COUSIN COÛTER COUVRENT COUVRIR CRAIGNAIT CAIGNEZ CRAINDRE CRAINS 
CRAINTE CRAINTES CRÉDIT CRÉDITS CRÉE CRÉENT CRÉER CRI CRIAIT CRIE CRIER CRIME CRIMES CRIS CRISE CRISES 
CRITIQUE CRITIQUES CROIRAIT CROIRE CROIS CROIT CROIX CROYAIENT CROYAIS CROYAIT CROYEZ CROYONS CRU 
CRURENT CRUT CUISINE CULTURE CULTURES CURÉ CURIEUSES CURIEUX CURIOSITÉ CURIOSITÉS D' DAME DAMES DANGER 
DANGEREUX DANGERS DANS DATE DATES DAVANTAGE DE DEBOUT DÉBUT DÉBUTS DÉCEMBRE DÉCIDÉ DÉCIDER DÉCIDÉS 
DÉCLARENT DÉCLARER DÉCOUVERT DÉCOUVRAIT DÉCOUVRIR DÉCOUVRIT DEDANS DÉFAUT DÉFAUTS DÉFEND DÉFENDAIT 
DÉFENDRE DÉFENDS DÉFENSE DÉFINITIF DEGRÉ DEGRÉS DEHORS DÉJÀ DÉJEUNER DÉLICAT DÉLICATES DÉLICATS 
DÉLICIEUX DEMAIN DEMANDAIENT DEMANDAIS DEMANDAIT DEMANDANT DEMANDE DEMANDÉ DEMANDENT DEMANDER DEMANDES 
DEMANDEZ DEMANDONS DEMEURAIT DEMEURENT DEMEURER DEMI DEMIE DENT DENTS DÉPART DÉPARTS DÉPASSAIT DÉPASSENT 
DÉPASSER DÉPEND DÉPENDENT DÉPENDRE DÉPENSER DÉPOSER DEPUIS DERNIER DERNIÈRE DERNIÈRES DERNIERS DERRIÈRE 
DES DÈS DESCEND DESCENDAIT DESCENDANT DESCENDENT DESCENDIT DESCENDRE DESCENDS DÉSERT DÉSESPOIR DÉSIR 
DÉSIRAIT DÉSIRER DÉSIREZ DÉSIRS DÉSORMAIS DESSOUS DESSUS DÉTAIL DÉTAILS DEUX DEUXIÈME DEVAIENT DEVAIS 
DEVAIT DEVANT DÉVELOPPÉE DÉVELOPPEMENT DÉVELOPPEMENTS DÉVELOPPENT DÉVELOPPER DEVENAIENT DEVENAIT DEVENIR 
DEVENU DEVENUE DEVENUES DEVENUS DEVEZ DEVIENNENT DEVIENT DEVINAIT DEVINER DEVINT DEVOIR DEVOIRS DEVONS 
DEVRAIENT DEVRAIS DEVRAIT DEVRIEZ DEVRONT DIABLE DIABLES DIEU DIEUX DIFFÉRENCE DIFFÉRENCES DIFFÉRENT 
DIFFÉRENTE DIFFÉRENTES DIFFÉRENTS DIFFICILE DIFFICILES DIFFICULTÉ DIFFICULTÉS DIGNE DIMANCHE DÎNER DIRAIS 
DIRAIT DIRE DIRECTEUR DIRECTION DIRECTIONS DIREZ DIRIGEAIT DIRIGER DIRONS DIS DISAIENT DISAIS DISAIT 
DISANT DISCOURS DISCUTER DISENT DISIEZ DISIONS DISONS DISPARAISSENT DISPARAÎT DISPARAÎTRE DISPARUT 
DISPOSAIT DISPOSER DISPOSITION DISPOSITIONS DISQUE DISTANCE DISTANCES DISTINGUENT DISTINGUER DIT DITES 
DIVERS DIVERSES DIVIN DIX DIX-HUIT DIX-NEUF DIX-SEPT DOCTEUR DOCTEURS DOIGT DOIGTS DOIS DOIT DOIVENT 
DOMAINE DOMAINES DOMESTIQUE DOMESTIQUES DONC DONNAIENT DONNAIT DONNANT DONNE DONNÉ DONNÉE DONNÉES DONNENT 
DONNER DONNERAIENT DONNERAIT DONNÉS DONNEZ DONNONS DONT DORMAIT DORMIR DORS DOS DOS À DOS DOUBLE DOUCE 
DOUCEMENT DOUCES DOULEUR DOULEURS DOUTAIT DOUTE DOUTER DOUX DROIT DROIT AU BUT DROITE DROITS DROITS ACQUIS 
DROITS ÉLÉVÉS DRÔLE DRÔLES DU DÛ DUC DUR DURENT DURER DURES DURS DUT EAU EAUX ÉCHAPPAIT ÉCHAPPENT ÉCHAPPER 
ÉCLAIRER ÉCLAT ÉCLATAIT ÉCLATER ÉCLATS ÉCOLE ÉCOLES ÉCONOMIE ÉCONOMIES ÉCONOMIQUE ÉCONOMIQUES ÉCOUTAIT 
ÉCOUTANT ÉCOUTE ÉCOUTER ÉCOUTEZ ÉCRIER S'ÉCRIER ÉCRIRE ÉCRIS ÉCRIT ÉCRITE ÉCRITES ÉCRITS ÉCRIVAIN ÉCRIVAINS 
ÉCRIVAIS ÉCRIVAIT ÉCRIVANT ÉCRIVENT ÉCRIVEZ ÉCRIVIT ÉDUCATION EFFET EFFET EN EFFETS EFFORT EFFORTS ÉGAL 
ÉGALEMENT ÉGARD ÉGARDS ÉGLISE ÉGLISES EH ÉLECTRIQUE ÉLECTRIQUES ÉLÉMENT ÉLÉMENTS ÉLEVÉ ÉLÈVE ÉLEVÉE ÉLEVÉES 
ÉLÈVENT ÉLEVER ÉLEVÉS ELLE -T-ELLE ELLES -ELLES ÉLOIGNER EMBRASSER ÉMISSION EMMENER ÉMOTION ÉMOTIONS EMPÊCHAIT 
EMPÊCHER EMPEREUR EMPIRE EMPLOIE EMPLOYANT EMPLOYER EMPORTANT EMPORTER EN -EN -T'EN ENCORE ENDROIT ENDROITS 
ÉNERGIE ÉNERGIES ENFANCE ENFANT ENFANTS ENFIN ENGAGER ENLEVER ENNEMI ENNUI ENNUIS ENNUYER ÉNORME ÉNORMES 
ENREGISTRER ENSEMBLE ENSEMBLES ENSUITE ENTEND ENTENDAIT ENTENDENT ENTENDEZ ENTENDIS ENTENDIT ENTENDONS 
ENTENDRE ENTENDS ENTENDU ENTENDUE ENTENDUES ENTENDUS ENTIER ENTIÈRE ENTIÈRES ENTIERS ENTOURAIT ENTOURER 
ENTRAÎNER ENTRAIT ENTRANT ENTRE ENTRÉE ENTRER ENTRÈRENT ENTRETENIR ENTRETIENT ENTREZ ENVIE ENVIRON ENVOIE 
ENVOYANT ENVOYER ÈPAULE ÈPAULES ÈPOQUE ÈPOQUES ÈPREUVE ÈPREUVES ÈPROUVAIT ÈPROUVER ÈQUIPE ÈQUIPES ERREUR 
ERREURS ES ESCALIERS ESPACE ESPÈCE ESPÈCES ESPÉRAIS ESPÉRAIT ESPÉRANCE ESPÉRANCES ESPÈRE ESPÈRER ESPOIR 
ESPOIRS ESPRIT ESPRITS ESSAIE ESSAYAIT ESSAYER ESSAYEZ EST ESTIMER ET ET CETERA ÉTABLIR ÉTABLIT ÉTAGE ÉTAGES 
ÉTAIENT ÉTAIS ÉTAIT ÉTANT ÉTAT ÉTATS ÉTÉ ÉTEND ÉTENDAIT ÉTENDRE ÉTERNEL ÊTES ÉTIEZ ÉTIONS ÉTOILE ÉTOILES 
ÉTONNAIT ÉTONNER ÉTONNERAIT ÉTRANGE ÉTRANGER ÉTRANGERS ÉTRANGES ÊTRE ÊTRES ÉTROIT ÉTUDE ÉTUDES ÉTUDIANT 
ÉTUDIANTS ÉTUDIER EU EUE EUES EURENT EUS EUSSENT EUT EÛT EUX ÉVÉNEMENT ÉVÉNEMENTS ÉVIDEMMENT ÉVITER EXACT 
EXACTEMENT EXAMINER EXCELLENT EXCELLENTS EXEMPLE EXEMPLES EXERCER EXIGEAIT EXIGER EXISTAIT EXISTANT EXISTE 
EXISTENCE EXISTENT EXISTER EXPÉRIENCE EXPÉRIENCES EXPLICATION EXPLICATIONS EXPLIQUE EXPLIQUER EXPOSER 
EXPRESSION EXPRESSIONS EXPRIMAIT EXPRIMER EXTÉRIEUR EXTÉRIEURES EXTÉRIEURS EXTRAORDINAIRE EXTRAORDINAIRES 
EXTRÊME EXTRÊMEMENT EXTRÊMES FACE FACES FACILE FACILEMENT FACILES FAÇON FAÇONS FACULTÉ FACULTÉS FAIBLE 
FAIBLES FAIBLESSE FAIM FAIRE FAIS FAISAIENT FAISAIS FAISAIT FAISANT FAISIONS FAISONS FAIT FAIT ACQUIS 
FAIT ET CAUSE FAITE FAITES FAITS FALLAIT FALLOIR FALLUT FAMEUX FAMILLE FAMILLES FATIGUER FAUDRA FAUDRAIT 
FAUSSES FAUT FAUTE FAUTES FAUX FAVEUR FAVEURS FEMME FEMMES FEMMES ET ENFANTS FEMMES ET HOMMES FENÊTRE 
FENÊTRES FER FERA FERAIS FERAIT FEREZ FERIEZ FERME FERMÉ FERMÉE FERMÉES FERMER FERMES FERMÉS FERONS FERONT 
FÊTE FÊTES FEU FEUILLE FEUILLES FEUX FÉVRIER FICHER FIDÈLE FIDÈLES FIER FIGURE FIGURENT FIGURER FIGURES 
FIGUREZ FIL FILLE FILLES FILM FILMS FILS FIN FINALEMENT FINES FINI FINIE FINIES FINIR FINIRAIT FINIS 
FINISSAIT FINISSENT FINIT FINS FIRENT FIS FIT FÎT FIXER FIXÉS FLAMME FLAMMES FLEUR FLEURS FLEUVE FLEUVES 
FLOT FLOTS FOI FOIE FOIS FOLIE FOLLES FONCTION FONCTIONS FOND DE FOND EN COMBLE FONT FORCE FORCÉMENT 
FORCES FORÊT FORÊETS FORMANT FORME FORMÉE FORMENT FORMER FORMES FORMIDABLE FORMULE FORMULES FORT FORTE 
FORTES FORTS FORTUNE FOU FOULE FOULES FOURNIR FOURNISSENT FOURNIT FOUS FOUTRE FRAÎCHES FRAIS FRANC 
FRANÇAIS FRANÇAISE FRANÇAISES FRANCS FRAPPÉ FRAPPÉE FRAPPÉES FRAPPER FRAPPÉS FRÈRE FRÈRES FROID FROIDES 
FROIDS FRONT FRONTIÈRE FRONTIÈRES FRUIT FRUITS FUIR FUMAIT FUMER FURENT FUS FUSSENT FUT FÛT GAGNAIT 
GAGNER GARÇON GARÇONS GARDAIT GARDE GARDENT GARDER GARDEZ GARE GARES GARS GAUCHE GAZ GÊNER GÉNÉRAL 
GÉNÉRALE GÉNÉRALES GÉNÉRAUX GÉNÉREUX GÉNIE GENOU GENOUX GENRE GENRES GENS GENTIL GESTE GESTES GLACE 
GLISSER GLOIRE GORGE GOSSE GOSSES GOÛT GOÛTS GOUVERNEMENT GOUVERNEMENTS GRÂCE GRAND GRAND-PÈRE GRANDE 
GRANDES GRANDEUR GRANDS GRAVE GRAVES GREC GRIS GROS GROS ET GRAS GROSSES GROUPE GROUPES GUÈRE GUERRE 
GUIDE HABILLER HABITAIT HABITENT HABITER HABITUDE HABITUDES HAINE HASARD HAUT DE HAUT EN BAS HAUTE 
HAUTES HAUTEUR HAUTS HEIN HÉLAS HERBE HERBES HÉSITER HEURE HEURES HEURES ET DEMI HEUREUSE HEUREUSEMENT 
HEUREUSES HEUREUX HIER HISTOIRE HISTOIRES HIVER HOMME HOMMES HOMMES ET FEMMES HONNÊTE HONNÊTES HONNEUR 
HONNEURS HONTE HONTEUX HÔPITAL HÔPITAUX HORIZON HORIZONS HORREUR HORS HÔTEL HÔTELS HUILE HUILES HUIT 
HUMAIN HUMAINE HUMAINES HUMAINS HUMANITÉ HUMEUR ICI IDÉE IDÉES IGNORAIT IGNORENT IGNORER IL -IL -T-IL 
ÎLE ÎLES ILLUSION ILLUSIONS -ILS ILS IMAGE IMAGES IMAGINAIS IMAGINATION IMAGINER IMAGINEZ IMMÉDIATEMENT 
IMMENSE IMMENSES IMMOBILE IMPORTANCE IMPORTANT IMPORTANTE IMPORTANTES IMPORTANTS IMPORTE IMPORTER IMPOSAIT 
IMPOSE IMPOSÉE IMPOSENT IMPOSER IMPOSSIBLE IMPOSSIBLES IMPRESSION IMPRESSIONS INCONNU INCONNUE INCONNUS 
INDIQUANT INDIQUER INDIVIDU INDIVIDUS INFÉRIEUR INFÉRIEURES INFÉRIEURS INFINI INFINIE INFIRMIER INFIRMIÈRE 
INFIRMIÈRES INFLUENCE INFLUENCES INITIATIVE INITIATIVES INQUIÉTUDE INSISTER INSPIRAIT INSPIRER INSTALLATION 
INSTALLATIONS INSTALLER INSTANT INSTANTS INSTINCT INSTINCTS INTELLIGENCE INTELLIGENT INTELLIGENTS INTENTION 
INTENTIONS INTÉRESSAIT INTÉRESSANT INTÉRESSANTES INTÉRESSENT INTÉRESSER INTÉRÊT INTÉRÊTS INTÉRIEUR INTERIEURES 
INTERNATIONAL INTERNATIONALES INTERNATIONAUX INTERROGER INTERROGEZ INTIME INTRODUIRE INTRODUIT INUTILE INUTILES 
INVITER ITALIEN ITALIENS J' JADIS JAMAIS JAMBE JANVIER JARDIN JARDINS JAUNE JAUNES JE -JE JETAIT JETANT 
JETER JEU JEUDI JEUNE JEUNE FILLE JEUNES JEUNESSE JEUX JOIE JOIES JOLI JOLIE JOLIES JOLIS JOUAIENT JOUAIT 
JOUE JOUÉE JOUENT JOUER JOUEZ JOUR JOURNAL JOURNAUX JOURNÉE JOURNÉES JOURS JUGEMENT JUGEMENTS JUGER JUILLET 
JUIN JUSQU' JUSQUE JUSTE JUSTEMENT JUSTES JUSTICE KILOMÈTRE KILOMÈTRES L' LA -LA LÀ -LÀ LÀ-BAS LÀ-DEDANS 
LÀ-DESSUS LÀ-HAUT LAINE LAISSAIENT LAISSAIT LAISSANT LAISSÂT LAISSE LAISSÉ LAISSÉE LAISSENT LAISSER 
LAISSÉS LAISSEZ LAISSONS LAIT LANCER LANGAGE LANGUE LANGUES LAQUELLE LARGE LARGES LARME LARMES LAVER 
LE -LE LEÇON LEÇONS LECTURE LÉGER LÉGÈRE LÉGERS LENDEMAIN LENTEMENT LEQUEL LES -LES LESQUELLES LESQUELS 
LETTRE LETTRES LEUR -LEUR LEURS LEVAIT LEVÉE LEVER LÈVRE LÈVRES LIBERTÉ LIBERTÉS LIBRE LIBRES LIEU LIEUX 
LIGNE LIGNES LIRE LIS LISAIT LIT LITS LITTÉRATURE LIVRE LIVRÉE LIVRER LIVRES LOGER LOI LOIN LOINTAIN 
LOINTAINES LOIS LONG LE LONG DE LONGS LONGTEMPS LONGUE LONGUES LORS LORSQU' LORSQUE LOUÉ LOUÉE LOUÉES 
LOUER LOUÉS LOURD LOURDES LOURDS LU LUE LUES LUI -LUI LUMIÈRE LUMIÈRES LUNDI LUNE LUS LUTTE LYCÉE M' MA 
MACHINE MACHINES MADAME MADEMOISELLE MAGASIN MAGASINS MAGNIFIQUE MAGNIFIQUES MAI MAIN MAINS MAINTENANT 
MAINTENIR MAINTIENT MAIS MAISON MAISONS MAÎTRE MAÎTRES MAÎTRESSE MAL MALADE MALADES MALADIE MALADIES 
MALGRÉ MALHEUR MALHEUREUSEMENT MALHEUREUX MAMAN MANGEAIT MANGER MANIÈRE MANIÈRES MANIFESTER MANQUAIT 
MANQUE MANQUENT MANQUER MARCHAIT MARCHAND MARCHANDS MARCHANT MARCHE MARCHÉ MARCHER MARCHÉS MARDI MARI 
MARIAGE MARIER MARQUENT MARQUER MARS MASSE MASSES MATÉRIEL MATÉRIELS MATIÈRE MATIÈRES MATIN MATINS 
MAUVAIS MAUVAISE MAUVAISES MAUX ME MÉCANIQUE MÉCANIQUES MÉDECIN MÉDECINE MÉDECINS MEILLEUR MEILLEURE 
MEILLEURES MEILLEURS MÊLER MEMBRE MEMBRES MÊME -MÊME -MÊMES MÊMES MÉMOIRE MÉNAGE MENAIT MENER MER 
MERCI MERCREDI MÈRE MÈRES MÉRITENT MÉRITER MERS MERVEILLEUSES MERVEILLEUX MES MESDAMES 
MESDAMES ET MESSIEURS MESDEMOISELLES MESSIEURS MESSIEURS ET MESDAMES MESURE MESURES MET MÉTHODE 
MÉTHODES MÉTIER MÉTIERS MÈTRE MÈTRES MÉTRO METS METTAIENT METTAIT METTANT METTENT METTEZ METTRE 
MEUBLE MEUBLES MEURENT MEURT MIDI MIEN MIENS MIEUX MILIEU MILIEUX MILITAIRE MILITAIRES MILLE MILLION 
MILLIONS MINISTÈRE MINISTÈRES MINISTRE MINISTRES MINUTE MINUTES MIRENT MIS MISE MISÉRABLE MISÈRE MISÈRES 
MISES MIT MODE MODERNE MODERNES MOI -MOI MOINS MOIS MOITIÉ MOMENT MOMENTS MON MONDE MONSIEUR MONTAGNE 
MONTAGNES MONTAIT MONTANT MONTENT MONTER MONTRAIENT MONTRAIT MONTRANT MONTRE MONTRÉ MONTRÉE MONTRENT 
MONTRER MORAL MORALES MORAUX MORCEAU MORCEAUX MORT MORTES MORTS MOT MOT À MOT MOTEUR MOTEURS MOTS MOURIR 
MOUVEMENT MOUVEMENTS MOYEN MOYENS MUR MURMURAIT MURMURER MURS MUSÉE MUSÉES MUSIQUE MYSTÈRE MYSTÈRES 
MYSTÉRIEUX N' NAÎT NAÎTRE NAQUIT NATION NATIONAL NATIONALE NATIONALES NATIONAUX NATIONS NATURE NATUREL 
NATURELLE NATURELLEMENT NATURELLES NATURELS NE NÉ NÉCESSAIRE NÉCESSAIRES NÉCESSITÉ NÉCESSITÉS NÉE NÉES 
NEIGE NÉS NET NETS NETTES NEUF NEUVES NEZ NI NOBLE NOBLES NOËL NOIR NOIRES NOIRS NOM NOMBRE NOMBRES 
NOMBREUSES NOMBREUX NOMMER NOMS NON NON PLUS NORD NORMAL NORMALES NORMAUX NOS NOTE NOTER NOTES NOTION 
NOTIONS NOTONS NOTRE LE NÔTRE LES NÔTRES NOUS -NOUS NOUVEAU NOUVEAUX NOUVEL NOUVELLE NOUVELLES NOVEMBRE 
NU NUAGE NUAGES NUE NUES NUIT NUIT ET JOUR NUIT ÉTOILÉE NUITS NUL NUMÉRO NUS OBJET OBJETS OBLIGÉ 
OBLIGEAIT OBLIGER OBLIGÉS OBSCUR OBSCURES OBSCURS OBSERVAIT OBSERVER OBTENIR OBTIENDRONT OBTIENT OBTINT 
OCCASION OCCASIONS OCCUPAIT OCCUPANT OCCUPÉ OCCUPÉE OCCUPÉES OCCUPENT OCCUPER OCCUPÉS OCTOBRE ODEUR OEIL 
OEUF OEUFS OEUVRE OEUVRES OFFERT OFFICIEL OFFICIELS OFFICIER OFFICIERS OFFRAIT OFFRANT OFFRE OFFRENT 
OFFRIR OH OISEAU OISEAUX OMBRE OMBRES ON -ON -T-ON ONCLE ONT ONZE OPÉRA OPÉRATION OPÉRATIONS OPÉRER 
OPINION OPINIONS OR ORDINAIRE ORDINAIRES ORDRE ORDRES OREILLE OREILLES ORGANE ORGANES ORGANISER ORGUEIL 
ORGUEILLEUX ORIGINE ORIGINES OSAIT OSER OU OÙ OUBLIAIS OUBLIAIT OUBLIE OUBLIÉ OUBLIÉE OUBLIÉES OUBLIER 
OUBLIÉS OUBLIEZ OUI OUTRE OUVERT OUVERTES OUVERTS OUVRAGE OUVRAGES OUVRAIT OUVRE OUVRENT OUVRIER OUVRIERS 
OUVRIR OUVRIT PAGE PAGES PAIE PAIENT PAIN PAIX PALAIS PÂLE PAPA PAPIER PAPIERS PAR PARAISSAIENT PARAISSAIT 
PARAISSANT PARAISSENT PARAÎT PARAÎTRE PARCE QUE PARDON PARDONNER PAREIL PAREILLES PAREILS PARENT PARENTS 
PARFAIT PARFAITEMENT PARFAITES PARFAITS PARFOIS PARISIEN PARLAIENT PARLAIS PARLAIT PARLANT PARLE PARLÉ 
PARLENT PARLER PARLES PARLEZ PARLIONS PARLONS PARMI PAROLE PAROLES PARS PART DE PART ET D'AUTRE PARTAGER 
PARTANT PARTENT PARTEZ PARTI PARTICULIER PARTICULIÈREMENT PARTICULIÈRES PARTICULIÈRS PARTIE PARTIES PARTIR 
PARTIS PARTIT PARTONS PARTOUT PARTS PARURENT PARUT PARVENAIT PARVENIR PAS PAS À PAS PAS DU TOUT PASSAGE 
PASSAGES PASSAIENT PASSAIT PASSANT PASSE PASSÉ PASSÉE PASSÉES PASSENT PASSER PASSÈRENT PASSÉS PASSEZ 
PASSION PASSIONS PASSONS PATRIE PATRON PATRONS PAUVRE PAUVRES PAYÉE PAYER PAYS PAYSAN PAYSANS PEAU 
PEINE À PEINE PEINES PEINTRE PEINTRES PEINTURE PEINTURES PENDANT PÉNÈTRENT PÉNÉTRER PÉNIBLE PENSAIS 
PENSAIT PENSANT PENSE PENSÉ PENSÉE PENSÉES PENSENT PENSER PENSES PENSEZ PENSONS PERD PERDAIT PERDIT 
PERDRE PERDS PERDU PERDUE PERDUS PÈRE PÈRES PÉRIODE PÉRIODES PERMET PERMETS PERMETTANT PERMETTENT 
PERMETTEZ PERMETTRAIT PERMETTRE PERMIS PERMIT PERSONNAGE PERSONNAGES PERSONNE PERSONNEL PERSONNELLES 
PERSONNELS PERSONNES PETIT PETITE PETITES PETITS PEU UN PEU PEUPLE PEUPLES PEUR PEUT PEUT-ÊTRE PEUVENT 
PEUX PHÉNOMÈNE PHÉNOMÈNES PHILOSOPHE PHILOSOPHES PHILOSOPHIE PHOTOGRAPHIE PHOTOGRAPHIES PHRASE PHRASES 
PIÈCE PIÈCES PIED DE PIED EN CAP PIEDS PIERRE PIERRES PITIÉ PLACE PLACÉ PLACER PLACES PLAINDRE PLAINT 
PLAIRE PLAIS PLAISAIT PLAISEZ PLAISIR PLAISIRS PLAÎT PLAN PLANS PLAT PLATS PLEIN PLEINE PLEINES PLEINS 
PLEURAIT PLEURER PLEUT PLEUVOIR PLUIE PLUIES PLUPART LA PLUPART PLUS PLUS OU MOINS PLUSIEURS PLUTÔT 
POCHE POCHES POÉSIE POÈTE POÈTES POINT POINTE POINTS POISSON POISSONS POITRINE POLICE POLITIQUE PONT 
PONTS POPULATION POPULATIONS PORT PORTAIENT PORTAIT PORTANT PORTE PORTÉ PORTENT PORTER PORTES PORTÉS 
PORTRAIT PORTRAITS PORTS POSÉE POSENT POSER POSITION POSITIONS POSSÉDAIT POSSÈDE POSSÈDENT POSSÉDER 
POSSÉDONS POSSIBLE POSSIBLES POSTE POSTES POT POT AUX ROSES POULE POULES POUR POUR QUE POURQUOI POURRA 
POURRAIENT POURRAIS POURRAIT POURREZ POURRIEZ POURRONS POURRONT POURSUIT POURSUIVAIT POURSUIVENT 
POURSUIVIE POURSUIVRE POURTANT POUSSAIT POUSSANT POUSSER POUSSIÈRE POUSSIÈRES POUVAIENT POUVAIT POUVANT 
POUVEZ POUVIONS POUVOIR POUVOIRS POUVONS PRATIQUE PRATIQUES PRÉCÉDENT PRÉCIEUSES PRÉCIEUX PRÉCIS 
PRÉCISÉMENT PRÉCISES PRÉFÉRAIS PRÉFÉRER PREMIER PREMIÈRE PREMIÈRES PREMIERS PRENAIT PRENANT PREND PRENDRE 
PRENDS PRENEZ PRENNENT PRENONS PRÉPARAIT PRÉPARER PRÈS PRÉSENCE PRÉSENT PRÉSENTAIT PRÉSENTANT PRÉSENTE 
PRÉSENTENT PRÉSENTER PRÉSENTES PRÉSENTS PRÉSIDENT PRESQUE PRÊT PRÊTAIT PRÉTEND PRÉTENDAIT PRÉTENDRE 
PRÉTENDS PRÊTER PRÊTES PRÊTS PREUVE PRÉVENIR PRÉVOIR PRÉVOIT PRIANT PRIE PRIER PRIÈRE PRIÈRES PRINCE 
PRINCES PRINCESSE PRINCIPAL PRINCIPALES PRINCIPAUX PRINCIPE PRINCIPES PRINTEMPS PRIRENT PRIS PRISES 
PRISON PRISONS PRIT PRIX PROBLÈME PROBLÈMES PROCÉDER PROCHAIN PRODIGIEUX PRODUCTION PRODUIRE PRODUISENT 
PRODUIT PRODUITES PRODUITS PROFESSEUR PROFESSEURS PROFITER PROFOND PROFONDE PROFONDES PROGRAMME PROGRÈS 
PROJET PROJETS PROMENADE PROMENADES PROMENER PROMETS PROMETTRE PROMIS PRONONCER PROPOS PROPOSER 
PROPOSITION PROPOSITIONS PROPRE PROPRES PROPRIÉTAIRE PROPTIÉTAIRES PROPRIÉTÉ PROPRIÉTÉS PROUVENT PROUVER 
PROVINCE PROVINCES PU PUBLIC PUBLICS PUBLIE PUBLIENT PUBLIER PUBLIQUES PUIS PUISQU' PUISQUE PUISSANCE 
PUISSANCES PUISSANT PUISSANTES PUISSANTS PUISSE PUISSENT PUISSIEZ PUR PURE PURES PURS PUT PÛT QU' 
QU'EST-CE QUI QUALITÉ QUALITÉS QUAND QUANT À QUANTITÉ QUANTITÉS QUARANTE QUART QUARTIER QUARTIERS 
QUARTS QUATORZE QUATRE QUE QUEL QUELLE QUELLES QUELQUE QUELQUE CHOSE QUELQUEFOIS QUELQUES QUELS 
QUESTION QUESTIONS QUI QUI EST-CE QUI QUINZE QUITTAIT QUITTÉ QUITTER QUOI QUOIQUE RACE RACES 
RACONTAIT RACONTER RACONTEZ RADIO RAISON RAMASSER RAMENAIT RAMENER RAPIDE RAPIDEMENT RAPIDES 
RAPPELAIT RAPPELANT RAPPELER RAPPELEZ RAPPELLE RAPPELLENT RAPPELONS RAPPORT RAPPORTENT RAPPORTER 
RAPPORTS RARE RARES RAYON RAYONS RÉALISER RÉALITÉ RECEVAIT RECEVOIR REÇOIS REÇOIT REÇOIVENT 
RECOMMENCER RECONNAIS RECONNAISSAIT RECONNAISSEZ RECONNAÎT RACONNAÎTRE RECONNUT REÇU REÇUE RECUEILLIR 
REÇUES REÇUS REÇUT RÉDUIRE RÉDUIT RÉEL RÉELLES RÉELS REFAIRE REFAIT RÉFLÉCHIR RÉFLÉCHIT RÉFLEXION 
REFUSAIT REFUSER REGARD REGARDAIS REGARDAIT REGARDANT REGARDE REGARDENT REGARDER REGARDEZ REGARDS 
RÉGIME RÉGIMES RÉGION RÉGIONS RÈGLE RÈGLES REGRETTER REINE RELATION RELATIONS RELÈVENT RELEVER 
RELIGIEUX RELIGION RELIGIONS REMARQUER REMARQUONS REMERCIE REMERCIER REMETTRE REMIS REMIT REMONTAIT 
REMONTER REMPLACER REMPLIR REMPLIT RENCONTRAIT RENCONTRE RENCONTRÉ RENCONTENT RENCONTRER RENCONTRES 
REND RENDAIENT RENDAIT RENDANT RENDENT RENDEZ RENDIT RENDRE RENDS RENDU RENDUE RENDUES RENDUS 
RENONCER RENTRANT RENTRER REPARTIR REPAS REPASSER RÉPÉTAIT RÉPÉTER RÉPOND RÉPONDAIT RÉPONDENT 
RÉPONDEZ RÉPONDIT RÉPONDRE RÉPONDS RÉPONSE RÉPONSES REPOS REPOSAIT REPOSER REPRENAIENT REPRENAIT 
REPRENANT REPREND REPRENDRE REPRENDS REPRENONS REPRÉSENTAIT REPRÉSENTANT REPRÉSENTE REPRÉSENTENT 
REPRÉSENTER REPRIS REPRIT REPROCHAIT REPROCHER RÉSERVE RÉSERVES RÉSOUDRE RESPECT RESPIRER 
RESPONSABILITÉ RESPONSABILITÉS RESSEMBLAIT RESSEMBLENT RESSEMBLER RESTAIENT RESTAIT RESTAURANT 
RESTE RESTÉ RESTÉE RESTÉES RESTENT RESTER RESTERAIT RESTÈRENT RESTES RESTÉS RESTEZ RESTONS RÉSULTAT 
RÉSULTATS RETARD RETENIR RETIENT RETIRER RETOUR RETOURNER RETRAITE RETROUVAIT RETROUVE RETROUVER 
RETROUVONS RÉUNIR RÉUNIT RÉUSSIR RÉUSSIT RÊVE RÉVEILLAIT RÉVEILLER RÉVÉLAIT RÉVÉLER REVENATIENT 
REVENAIT REVENANT REVENIR REVENONS REVENU REVENUE REVENUES REVENUS RÊVER RÊVES REVIENDRONS 
REVIENNENT REVIENS REVIENT REVINRENT REVINT REVOIR RÉVOLUTION REVUE REVUES RIAIT RIANT RICHE 
RICHES RICHESSE RICHESSES RIDICULE RIEN RIEN DU TOUT RIEZ RIRE RISQUE RISQUER RIT ROBE ROBES 
ROI ROIS RÔLE RÔLES ROMAN ROMANS ROSE ROSES ROUGE ROUGES ROULER ROUTE ROUTES RUE RUES RUSSE 
RUSSES S' SA SAC SACHANT SACRÉ SACRÉE SACRÉES SACS SAGE SAINT SAINTS SAIS SAISIR SAISIT SAISON 
SAISONS SAIT SALLE SALLES SALON SAMEDI SANG SANS SANTÉ SAUF SAURAIENT SAURAIS SAURAIT SAURIONS 
SAUTAIT SAUTER SAUVAGE SAUVAGES SAUVER SAVAIENT SAVAIS SAVAIT SAVANT SAVANTS SAVENT SAVEZ SAVIEZ 
SAVOIR SAVONS SCÈNE SCÈNES SCIENCE SCIENCES SE SEC SECOND SECONDES SECONDS SECRET SECRÈTES SECRETS 
SECS SEIGNEUR SEIGNEURS SEIN SEIZE SELON SEMAINE SEMAINES SEMBLABLE SEMBLABLES SEMBLAIENT SEMBLAIT 
SEMBLE SEMBLENT SEMBLER SEMBLERAIT SENS SENSATION SENSATIONS SENSIBLE SENSIBLES SENT SENTAIS SENTAIT 
SENTANT SENTENT SENTEZ SENTI SENTIE SENTIES SENTIMENT SENTIMENTS SENTIR SENTIS SENTIT SÉPARAIT 
SÉPARER SEPT SEPTEMBRE SERA SERAI SERAIENT SERAIS SERAIT SERAS SEREZ SÉRIE SÉRIES SÉRIEUX SERIEZ 
SERIONS SERONS SERONT SERRER SERRÉS SERT SERVAIT SERVANT SERVENT SERVICE SERVICES SERVIR SES SEUL 
SEULE SEULEMENT SEULES SEULS SI SIÈCLE SIÈCLES SIGNALER SIGNE SIGNES SILENCE SIMPLE SIMPLEMENT 
SIMPLES SINGULIER SINON SITUATION SITUATIONS SIX SOCIAL SOCIALES SOCIAUX SOCIÉTÉ SOCIÉTÉS SOEUR 
SOEURS SOI SOIENT SOIGNER SOIN SOINS SOIR SOIRÉE SOIRÉES SOIRS SOIS SOIT SOIXANTE SOL SOLDAT 
SOLDATS SOLEIL SOLITUDE SOLUTION SOLUTIONS SOMBRE SOMBRES SOMME SOMMEIL SOMMES SON SONGEAIT 
SONGEANT SONGER SONGEZ SONNER SONS SONT SORS SORT SORTAIT SORTANT SORTE SORTENT SORTES SORTIE 
SORTIR SORTIT SOUCI SOUCIS SOUDAIN SOUFFERT SOUFFLE SOUFFRAIT SOUFFRANCE SOUFFRANCES SOUFFREZ 
SOUFFRIR SOUHAITAIT SOUHAITER SOUHAITONS SOURCE SOURCES SOURIAIT SOURIANT SOURIRE SOURIT SOUS 
SOUTENIR SOUTIENT SOUVENEZ SOUVENIR SE SOUVENIR SOUVENIRS SOUVENT SOUVIENS SOUVIENT SOYEZ SOYONS 
SPÉCIAL SPÉCIALES SPÉCIAUX SPECTACLE SPIRITUEL STATUE STATUES STYLE SU SUBIR SUBISSENT SUCCÈS 
SUFFI SUFFIRAIT SUFFIRE SUFFISAIT SUFFIT SUIS SUISSE SUIT SUITE SUITES SUIVAIT SUIVANT SUIVANTES 
SUIVANTS SUIVENT SUIVI SUIVIE SUIVIES SUIVIRENT SUIVIS SUIVIT SUIVRE SUJET SUJETS SUPÉRIEUR 
SUPÉRIEURE SUPÉRIEURES SUPÉRIEURS SUPPORTENT SUPPORTER SUPPOSER SUPPOSEZ SUPPOSONS SUR SÛR SÛRE 
SÛREMENT SURPRISE SURPRISES SURTOUT SURVEILLER SUT SYMPATHIQUE SYSTÈME SYSTÈMES T' TA TABLE TABLEAU 
TABLEAUX TABLES TAIRE TAIS TALENT TANDIS QUE TANT TANT ET PLUS TANT ET TANT TANT QUE TANTE TANTÔT 
TAPER TARD TAS TE TECHNIQUE TEL TÉLÉPHONE TÉLÉPHONER TELLE TELLEMENT TELLES TELS TÉMOIN TÉMOINS 
TEMPS DE TEMPS À AUTRE DE TEMPS EN TEMPS TENAIENT TENAIS TENAIT TENANT TEND TENDANT TENDENT TENDIT 
TENDRE TENDRESSE TENEZ TENIR TENONS TENTER TENU TENUE TENUS TERME TERMES TERMINER TERRAIN TERRAINS 
TERRE TERRES TERRIBLE TERRIBLES TES TÊTE TÊTES TEXTE TEXTES THÉÂTRE THÉÂTRES TIENNENT TIENS TIENT 
TIRAIT TIRER TISSU TITRE TITRES TOI -TOI TOMBAIT TOMBÉE TOMBENT TOMBER TOMBÉS TON TONS TORT TÔT 
TÔT OU TARD TOUCHENT TOUCHER TOUJOURS TOUR TOURNAIT TOURNANT TOURNENT TOURNER TOURS TOUS TOUS 
AZIMUTS À TOUS ÉGARDS TOUT TOUT À FAIT TOUT À L'HEURE TOUT D'UN COUP TOUT DE MÊME TOUT DE SUITE 
TOUT LE MONDE TOUTE TOUTEFOIS TOUTES TRADITION TRADITIONS TRAIN TRAINS TRAIT TRAITÉE TRAITER TRAITS 
TRANQUILLE TRANQUILLES TRANSFORMENT TRANSFORMER TRAVAIL TRAVAILLAIT TRAVAILLENT TRAVAILLER TRAVAILLONS 
TRAVAUX TRAVERS À TRAVERS TRAVERSANT TRAVERSENT TRAVERSER TREIZE TREMBLER TRENTE TRÈS TRISTE TRISTES 
TRISTESSE TROIS TROISIÈME TROMPÉE TROMPER TROMPEZ TROP TROU TROUPE TROUPES TROUVAIENT TROUVAIS TROUVAIT 
TROUVANT TROUVE TROUVÉ TROUVÉE TROUVENT TROUVER TROUVERAIT TROUVEREZ TROUVERONS TROUVES TROUVEZ TROUVONS 
TRUC TRUCS TU -TU TUE TUER TYPE TYPES UN UNE UNES LES UNES ET LES AUTRES UNIQUE UNIQUEMENT UNITÉ UNITÉS 
UNIVERS UNIVERSEL UNS LES UNS ET LES AUTRES USAGE USINE USINES UTILE UTILES VA VACANCES VAGUE VAGUES VAIN 
VAINE VAINES VAINS VAIS VALAIT VALENT VALEUR VALEURS VALOIR VAS VASTE VASTES VAUDRAIT VAUT VEILLE VENAIENT 
VENAIS VENAIT VENANT VENDRE VENDREDI VENEZ VENIR VENONS VENT VENTS VENU VENUE VENUES VENUS VÉRITABLE 
VÉRITABLES VÉRITÉ VÉRITÉS VERRAS VERRE VERRES VERREZ VERRONS VERS VERT VERTES VERTS VERTU VERTUS VEULENT 
VEUT VEUX VICTIME VICTIMES VICTOIRE VIDE VIDES VIE VIEIL VIEILLARD VIEILLARDS VIEILLE VIEILLES VIENDRAIT 
VIENNENT VIENS VIENT VIERGE VIES VIEUX VIF VIFS VIGOUREUX VILLAGE VILLAGES VILLE VILLES VIN VINGT VINGTS 
VINRENT VINS VINT VIOLENT VIOLENTE VIOLENTES VIOLENTS VIRENT VIS VISAGE VISAGES VISITE VISITER VISITES VIT 
VITE VITESSE VITESSES VIVAIENT VIVAIT VIVANT VIVANTES VIVANTS VIVENT VIVES VIVONS VIVRE VOICI VOIE VOIENT 
VOIES VOILÀ VOILE VOIR VOIS VOISIN VOISINS VOIT VOITURE VOITURES VOIX VOLONTÉ VOLONTÉS VOLUME VOLUMES VONT 
VOS VOTRE VOUDRAIS VOUDRAIT VOUDRAS VOUDREZ VOULAIENT VOULAIS VOULAIT VOULANT VOULEZ VOULOIR VOULONS VOULU 
VOULUT VOUS -VOUS VOUS ET MOI VOYAGE VOYAGES VOYAIENT VOYAIS VOYAIT VOYANT VOYEZ VOYONS VRAI VRAIE VRAIES 
VRAIMENT VRAIS VU VUE VUES VUS Y -Y YEUX
""".lower().split())
################################################################################
LANGDATA = dict(
	en=dict(
		syllables=countsyllables_en,
		words=words_en,
		beginnings=beginnings_en,
		basicwords=basicwords_en),
	nl=dict(
		syllables=countsyllables_nlde,
		words=words_nl,
		beginnings=beginnings_nl,
		basicwords=basicwords_nl),
	de=dict(
		syllables=countsyllables_nlde,
		words=words_de,
		beginnings=beginnings_de,
		basicwords=basicwords_de),
	# Settings for when the input language is French:
	fr=dict(
		syllables=count_syllables_fr,
		words=words_fr,
		beginnings=beginnings_fr,
		basicwords=basicwords_fr),
)
