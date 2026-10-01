"""Rule-checked synthetic contrasts; never presented as human annotations.

Template identities are disjoint across splits. Correct/incorrect editing cases
share a context within a split, so evaluation uncertainty must group by context.
"""
import hashlib
import json
import random
from pathlib import Path

NAMES = ['Mia', 'Tom', 'Lily', 'Ben', 'Ana', 'Sam', 'Lucy', 'Max', 'Ella', 'Leo',
         'Nora', 'Jack', 'Zoe', 'Alex', 'Rose', 'Finn']
COLORS = ['blue', 'red', 'green', 'yellow', 'pink', 'white', 'black', 'purple']
OBJECTS = ['hat', 'cup', 'ball', 'bag', 'box', 'coat', 'shirt', 'kite']
FILLERS = ['', ' It was a quiet day.', ' A little bird sang nearby.',
           ' The sun was shining.', ' A friend was waiting.', ' It was early in the morning.']
CATEGORIES = ['color', 'location', 'emotion', 'state', 'identity', 'agreement']


def render(category, template, rng):
    name = rng.choice(NAMES)
    if category == 'color':
        good = rng.choice(COLORS); obj = rng.choice(OBJECTS)
        wrong = rng.sample([c for c in COLORS if c != good], 3)
        facts = [f'{name} owned a {good} {obj}.', f'The {obj} was painted {good}.',
                 f'{name} chose the {good} {obj}.']
        middles = [f'{name} picked up the [X] {obj}.', f'The [X] {obj} belonged to {name}.',
                   f'{name} held the same [X] {obj}.', f'Later, {name} carried the [X] {obj}.']
        text = facts[template % 3] + ' ' + middles[template // 3] + f' Its color had not changed.'
        accepted = [good]
    elif category == 'location':
        good = rng.choice(['under', 'behind', 'beside', 'near'])
        wrong = rng.sample([c for c in ['under', 'behind', 'beside', 'near', 'above'] if c != good], 3)
        obj = rng.choice(['bed', 'chair', 'table', 'tree', 'bench', 'desk'])
        item = rng.choice(['toy', 'ball', 'box', 'bag', 'shoe'])
        facts = [f'{name} left the {item} {good} the {obj}.',
                 f'The {item} lay {good} the {obj}.', f'{name} placed a {item} {good} the {obj}.']
        middles = [f'Later, {name} found it [X] the {obj}.', f'The {item} was still [X] the {obj}.',
                   f'{name} looked [X] the {obj} for the {item}.', f'That {item} remained [X] the {obj}.']
        text = facts[template % 3] + ' ' + middles[template // 3] + ' Nobody had moved it.'
        accepted = [good]
    elif category in ('emotion', 'state'):
        options = [
            ('lost a favorite toy', ['sad', 'upset'], ['happy', 'excited', 'proud'], 'Tears ran down the little face.'),
            ('won a fun game', ['happy', 'excited'], ['sad', 'angry', 'afraid'], 'There was a big smile.'),
            ('heard a scary noise', ['scared', 'afraid'], ['calm', 'happy', 'brave'], 'Both hands shook with fear.'),
            ('had not eaten all day', ['hungry'], ['full', 'sleepy', 'cold'], 'The empty stomach growled.'),
            ('played all day without resting', ['tired', 'sleepy'], ['awake', 'hungry', 'excited'], 'It was time to rest.'),
        ] if category == 'emotion' else [
            ('got caught in the rain', ['wet'], ['dry', 'hot', 'clean'], 'Water dripped from the coat.'),
            ('sat by a warm fire', ['warm'], ['cold', 'wet', 'frozen'], 'The fire gave off plenty of heat.'),
            ('turned off the only lamp', ['dark'], ['bright', 'warm', 'wet'], 'There was no light in the room.'),
            ('emptied a box completely', ['empty'], ['full', 'heavy', 'wet'], 'There was nothing left inside.'),
            ('filled a box to the top', ['full'], ['empty', 'wet', 'broken'], 'There was no room for anything else.'),
        ]
        event, accepted, wrong, evidence = rng.choice(options)
        beginnings = [f'{name} {event}.', f'One day, {name} {event}.', f'After lunch, {name} {event}.']
        subject = name if category == 'emotion' or accepted[0] in ('wet', 'warm') else ('The room' if accepted[0] == 'dark' else 'The box')
        middles = [f'{subject} felt [X].', f'Now {subject.lower() if subject.startswith("The") else subject} was [X].',
                   f'{subject} was very [X].', f'At that moment, {subject.lower() if subject.startswith("The") else subject} seemed [X].']
        text = beginnings[template % 3] + ' ' + middles[template // 3] + ' ' + evidence
        # "box felt full" is unnatural: use a grammatical state predicate.
        if category == 'state':
            text = text.replace('felt [X]', 'was [X]')
    elif category == 'identity':
        gender = rng.choice(['girl', 'boy', 'children'])
        good = {'girl': 'She', 'boy': 'He', 'children': 'They'}[gender]
        wrong = [p for p in ['She', 'He', 'They', 'It'] if p != good]
        facts = [f'{name} was a little {gender}.', f'A {gender} named {name} came by.', f'{name}, a {gender}, went out.'] if gender != 'children' else [
            f'{name} and a friend were children.', 'Two children went out together.', f'{name} and two friends came by.']
        tails = ['[X] smiled and waved.', '[X] wanted to play a game.', '[X] walked to the park.', '[X] found a little ball.']
        text = facts[template % 3] + ' ' + tails[template // 3]
        accepted = [good]
    else:
        kind = rng.randrange(4)
        if kind == 0:
            good, bad = rng.choice([('walked','walk'), ('played','play'), ('went','go'), ('ate','eat'), ('ran','run'), ('saw','see')])
            starts = ['Yesterday,', 'Last week,', 'Last night,']
            tails = ['in the garden.', 'near the house.', 'with a friend.', 'before going home.']
            text = f'{starts[template % 3]} {name} [X] {tails[template // 3]}'
            wrong = [bad, 'is', 'are']; accepted = [good]
        elif kind == 1:
            good, bad = rng.choice([('walks','walk'), ('plays','play'), ('eats','eat'), ('runs','run'), ('goes','go')])
            starts = ['Every morning,', 'Each day,', 'Every afternoon,']
            tails = ['before school.', 'near the house.', 'with a friend.', 'in the park.']
            text = f'{starts[template % 3]} {name} [X] {tails[template // 3]}'
            accepted = [good]; wrong = [bad, 'were', 'are']
        elif kind == 2:
            plural = rng.choice(['dogs', 'cats', 'children', 'birds', 'puppies', 'boys'])
            starts = [f'The two {plural}', f'All three {plural}', f'Those {plural}']
            tails = ['playing together.', 'happy to be there.', 'waiting outside.', 'very tired.']
            text = f'{starts[template % 3]} [X] {tails[template // 3]}'
            accepted = ['were']; wrong = ['was', 'is', 'am']
        else:
            vowel = rng.choice(['apple', 'egg', 'orange', 'umbrella', 'elephant'])
            starts = [f'{name} saw', f'{name} found', f'{name} drew']
            tails = ['yesterday.', 'in a picture.', 'this morning.', 'for a friend.']
            text = f'{starts[template % 3]} [X] {vowel} {tails[template // 3]}'
            accepted = ['an']; wrong = ['a', 'two', 'many']
    # Same distractors across split identities; no category token is fed to ranker.
    text += rng.choice(FILLERS)
    prefix, suffix = text.split('[X]')
    return prefix, suffix, accepted, wrong


def alternatives(prefix,suffix,category,words):
    """Bounded explicit equivalences; NOT an open-ended semantic verifier."""
    values=list(words)
    if category in ('emotion','state') and not prefix.rstrip().endswith('very'):
        values += [modifier+word for word in words for modifier in ('very ','so ')]
    if category=='location':
        for word in words:
            values += {'under':['underneath','beneath'],'behind':['just behind','close behind'],
                       'beside':['next to'],'near':['close to']}[word]
    return list(dict.fromkeys(values))


def build(output, train=12000, validation=1200, test_contexts=300, seed=173, version=1):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh dataset directory; frozen data must not be overwritten')
    output.mkdir(parents=True)
    rng = random.Random(seed); seen = set(); manifest = {'version': version, 'seed': seed,
        'provenance': 'Procedural rule-checked synthetic contrasts; no model teacher; no human labels.',
        'license': 'Apache-2.0', 'splits': {}}
    for split, count, templates in [('train', train, list(range(8))), ('validation', validation, [8,9]), ('test', test_contexts, [10,11])]:
        contexts=[]; tries=0
        while len(contexts)<count:
            tries+=1
            if tries>count*100: raise ValueError('Requested too many unique contexts')
            category=CATEGORIES[len(contexts)%len(CATEGORIES)]
            template=rng.choice(templates)
            prefix,suffix,accepted,wrong=render(category,template,rng)
            if version>=2:accepted=alternatives(prefix,suffix,category,accepted)
            key=hashlib.sha256((prefix+'[X]'+suffix).encode()).hexdigest()
            if key in seen: continue
            seen.add(key)
            contexts.append({'id':f'{split}-{len(contexts):05}', 'context_sha256':key,
                'category':category,'template':template,'prefix':prefix,'suffix':suffix,
                'accepted':accepted,'negative':wrong})
        path=output/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in contexts),encoding='utf-8')
        manifest['splits'][split]={'contexts':len(contexts),'templates':templates,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def editing_cases(contexts):
    """Balanced matched clean/error inputs; accepted list is deliberately bounded."""
    for c in contexts:
        for clean in (False, True):
            fragment = c['accepted'][0] if clean else c['negative'][0]
            text=c['prefix']+fragment+c['suffix']; start=len(c['prefix'])
            yield {**c,'id':c['id']+('-clean' if clean else '-error'), 'clean':clean,
                   'text':text,'start':start,'end':start+len(fragment)}
