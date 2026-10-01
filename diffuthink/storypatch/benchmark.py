"""Frozen, authored challenge cases: useful probes, not a general coherence score."""
import hashlib
import json
from pathlib import Path

# Each case marks the WRONG passage to edit. Accepted answers are deliberately
# incomplete: a valid alternative can be missed by exact normalized matching.
CASES = [
    ('emotion', 'Lily lost her favorite toy. She felt [happy]. Tears ran down her face.', ['sad', 'very sad', 'upset']),
    ('emotion', 'Ben won the game. He felt [sad]. He smiled and cheered.', ['happy', 'very happy', 'excited', 'proud']),
    ('emotion', 'A loud crash woke Mia. She felt [calm]. Her hands shook with fear.', ['scared', 'afraid', 'frightened']),
    ('emotion', 'Sam had not eaten all day. He felt [full]. His stomach growled.', ['hungry', 'very hungry']),
    ('emotion', 'The baby had played for hours. She felt [awake]. She yawned and closed her eyes.', ['tired', 'sleepy', 'very tired']),
    ('emotion', 'Dad came home with a puppy. Ana was [angry]. She smiled and hugged her new pet.', ['happy', 'excited', 'very happy']),
    ('object', 'Mia had a blue cup. She drank from her [red] cup. It was the same blue cup.', ['blue']),
    ('object', 'Ben put on his red hat. His [green] hat kept his head warm. He loved its red color.', ['red']),
    ('object', 'The dog chased a ball. It picked up the [book] in its mouth and brought the ball back.', ['ball']),
    ('object', 'Lucy planted a seed. She watered the [stone] every day until the seed began to grow.', ['seed']),
    ('object', 'Tom opened his book. He read the [shoe] until he reached the last page.', ['book', 'story']),
    ('object', 'Mom baked a cake. She cut the [chair] into slices and served the cake.', ['cake']),
    ('state', 'The box was empty. There was [something] inside. It contained nothing at all.', ['nothing']),
    ('state', 'The rain soaked his coat. His coat was [dry]. Water dripped from it.', ['wet', 'very wet']),
    ('state', 'The fire went out. The room grew [hot]. Everyone needed a blanket.', ['cold', 'colder']),
    ('state', 'Dad turned off the only lamp. The room became [bright]. Nobody could see in the dark.', ['dark']),
    ('state', 'The door was locked. Mia could [open] it. She needed the key first.', ['not open', 'not unlock']),
    ('state', 'Lucy gave away all her cookies. She had [many] cookies left. Her plate was empty.', ['no', 'zero']),
    ('location', 'The children stayed inside the house. They played [outside] all afternoon without leaving the house.', ['inside', 'indoors']),
    ('location', 'Tom put his toy under the bed. Later he found it [on] the bed, exactly where he had left it.', ['under', 'underneath', 'beneath']),
    ('location', 'The bird sat in its nest. It stayed in the [pond] all night and left the nest at dawn.', ['nest']),
    ('location', 'The fish lived in a pond. It swam in the [sky] while the other fish watched.', ['pond', 'water']),
    ('location', 'Mia hid the key in her pocket. She reached into her [shoe] and took the key out of her pocket.', ['pocket']),
    ('location', 'The children sat at a table. Their plates were on the [floor]. They ate from the table.', ['table']),
    ('identity', 'Lily was a little girl. [He] put on her coat and went out.', ['She']),
    ('identity', 'Tom was a little boy. [She] picked up his toy and smiled.', ['He']),
    ('identity', 'Two girls went to the park. [She] played together until it was time to go home.', ['They']),
    ('identity', 'The puppy was a small dog. The [cat] wagged its tail and barked.', ['dog', 'puppy']),
    ('identity', 'Mia and Ben were friends. [He] both liked the same game.', ['They']),
    ('identity', 'The bird built a nest. The [fish] laid an egg in its nest.', ['bird']),
    ('agreement', 'Yesterday, Tom [walk] to the park. He stayed there all afternoon.', ['walked']),
    ('agreement', 'Every morning, she [eat] her breakfast before school.', ['eats']),
    ('agreement', 'There [was] three apples on the table.', ['were', 'are']),
    ('agreement', 'The two dogs [was] playing together in the garden.', ['were']),
    ('agreement', 'Yesterday, Mia [go] home after school.', ['went']),
    ('agreement', 'He gave me [a] apple. I ate the apple.', ['an']),
]


def cases():
    rows = []
    for index, (category, template, answers) in enumerate(CASES):
        start, end = template.index('['), template.index(']')
        text = template.replace('[', '').replace(']', '')
        rows.append({'id': f'challenge-{index:03}', 'category': category, 'text': text,
                     'start': start, 'end': end-1, 'accepted': answers})
    return rows


def freeze(path):
    path = Path(path)
    payload = json.dumps({'version': 1, 'cases': cases(),
        'limitations': 'Authored synthetic probes, no training use; exact answers are incomplete; no human rating.'}, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != payload:
        raise ValueError('Frozen benchmark differs: use a new version')
    path.write_text(payload, encoding='utf-8')
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == '__main__':
    print(freeze('reports/storypatch-v2/challenges.json'))
