/** Find recorded measurements without confusing the giver and receiver. */

/** @param {string} value */
function words(value) {
  return value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

const SINGULARS = new Map(
  /** @type {[string,string][]} */ ([
    ["abilities", "ability"],
    ["opportunities", "opportunity"],
    ["losses", "loss"],
    ["victories", "victory"],
    ["statuses", "status"],
    ...[
      "aura",
      "draw",
      "period",
      "reason",
      "return",
      "reward",
      "score",
      "win",
      "activation",
      "agent",
      "application",
      "assist",
      "attack",
      "break",
      "contribution",
      "count",
      "death",
      "effect",
      "fraction",
      "heal",
      "kill",
      "recipient",
      "rescue",
      "save",
      "share",
      "step",
      "target",
      "team",
      "tick",
      "trap",
      "use",
      "wave",
    ].map((word) => [`${word}s`, word]),
  ]),
);

/** @param {string} value */
function tokens(value) {
  return words(value)
    .split(" ")
    .filter(Boolean)
    .map((word) => SINGULARS.get(word) ?? word);
}

/** Only word beginnings match. Agent IDs and short words must match exactly.
 * @param {string} part @param {string} word
 */
function prefix(part, word) {
  return (
    part === word ||
    (part.length >= 3 &&
      !FILLER.has(part) &&
      !RELATIONS.has(part) &&
      !ORIENTATIONS.has(part) &&
      word.startsWith(part))
  );
}

const EQUIVALENTS = [
  ["damage taken", "damage received"],
  ["damage suffered", "damage received"],
  ["incoming damage", "damage received"],
  ["damage dealt", "damage done"],
  ["damage inflicted", "damage done"],
  ["outgoing damage", "damage done"],
  ["healing taken", "healing received"],
  ["incoming healing", "healing received"],
  ["healing given", "healing done"],
  ["healing provided", "healing done"],
  ["outgoing healing", "healing done"],
  ["not accepted", "rejected"],
];

/** @param {string[]} input */
function equivalents(input) {
  const output = [...input];
  for (const [phrase, replacement] of EQUIVALENTS) {
    const match = tokens(phrase);
    for (let i = 0; i <= output.length - match.length; i++) {
      if (match.every((word, j) => prefix(output[i + j], word))) {
        output.splice(i, match.length, ...tokens(replacement));
      }
    }
  }
  return output;
}

const FILLER = new Set([
  "a",
  "an",
  "the",
  "of",
  "for",
  "about",
  "is",
  "are",
  "was",
  "were",
  "this",
  "that",
  "these",
  "those",
  "all",
  "total",
  "overall",
  "id",
  "count",
]);
const RELATIONS = new Set(["to", "from", "by", "on", "against", "and", "between"]);
const ORIENTATIONS = new Set([
  "received",
  "done",
  "receiving",
  "given",
  "dealt",
  "taken",
]);

// Longer phrases take priority over a word inside them, such as Damage in Rescue.
const MEASURES = [
  {
    phrases: [
      "lethal damage rescue",
      "priest healing save",
      "healing save",
      "life saving healing",
      "save",
      "rescue",
    ],
    kinds: ["rescue"],
  },
  {
    phrases: [
      "damage prevented",
      "damage blocked",
      "damage added",
      "damage amplification",
      "damage mitigation",
      "aura benefit",
    ],
    kinds: ["aura_benefit"],
  },
  {
    phrases: [
      "priest healing prevented",
      "healing prevented",
      "healing prevention",
      "anti heal",
    ],
    kinds: ["poison_prevention"],
  },
  {
    phrases: [
      "regenerated healing",
      "automatic regeneration",
      "regeneration",
      "regen",
      "natural health recovery",
      "passive healing",
    ],
    kinds: ["regeneration"],
  },
  { phrases: ["excess healing", "overhealing", "overheal"], kinds: ["excess_healing"] },
  { phrases: ["effective healing", "useful healing"], kinds: ["effective_healing"] },
  {
    phrases: ["trap break", "broken trap", "observed trap period"],
    kinds: ["trap_break"],
  },
  {
    phrases: ["aura coverage", "aura covered", "aura eligibility"],
    kinds: ["aura_coverage"],
  },
  {
    phrases: ["status application", "effect application"],
    kinds: ["status_application"],
  },
  {
    phrases: [
      "status active step",
      "time with status",
      "time under status",
      "status duration",
    ],
    kinds: ["status_time"],
  },
  {
    phrases: ["freedom against slow", "freedom protection", "movement floor"],
    kinds: ["freedom"],
  },
  {
    phrases: ["respawn", "respawning", "waiting time", "wait time"],
    kinds: ["respawn"],
  },
  {
    phrases: ["focus fire", "coordination", "multi attacker tick"],
    kinds: ["coordination"],
  },
  { phrases: ["formation", "distance", "spread out"], kinds: ["formation"] },
  {
    phrases: [
      "ability activation",
      "ability use",
      "use",
      "activation",
      "application",
      "casting",
      "cast",
    ],
    kinds: ["activation"],
  },
  { phrases: ["action", "rejected action", "accepted action"], kinds: ["action"] },
  { phrases: ["death", "died", "dead"], kinds: ["death"] },
  { phrases: ["kill", "assist"], kinds: ["kill"] },
  { phrases: ["damage", "dmg"], kinds: ["damage"] },
  {
    phrases: ["healing", "heal", "health restored"],
    kinds: ["healing", "effective_healing", "excess_healing", "regeneration"],
  },
  { phrases: ["reward", "return"], kinds: ["return"] },
  { phrases: ["score"], kinds: ["score"] },
  { phrases: ["win", "victory"], kinds: ["outcome"], qualifier: "win" },
  { phrases: ["loss", "defeat"], kinds: ["outcome"], qualifier: "loss" },
  { phrases: ["draw", "tie"], kinds: ["outcome"], qualifier: "draw" },
  { phrases: ["outcome"], kinds: ["outcome"] },
  { phrases: ["episode length", "episode duration"], kinds: ["episode"] },
];

/** @param {string[]} input @param {string} phrase */
function phraseAt(input, phrase) {
  const match = tokens(phrase);
  return input.findIndex(
    (_, i) =>
      i + match.length <= input.length &&
      match.every((word, j) => prefix(input[i + j], word)),
  );
}

/** @param {string[]} input @param {string[]} phrases @param {boolean} [boundary] */
function takePhrase(input, phrases, boundary = false) {
  for (const phrase of phrases) {
    const start = phraseAt(input, phrase);
    if (start >= 0) {
      input.splice(
        start,
        tokens(phrase).length,
        ...(boundary ? ["measure-boundary"] : []),
      );
      return true;
    }
  }
  return false;
}

const EFFECTS = [
  {
    phrases: ["hunter basic slow", "hunter slow", "serrated arrow"],
    status: "hunter_basic_slow",
  },
  { phrases: ["warrior charge slow", "charge slow"], status: "warrior_charge_slow" },
  { phrases: ["warrior charge stun", "charge stun"], status: "warrior_charge_stun" },
  { phrases: ["rogue poison slow", "poison slow"], status: "rogue_poison_slow" },
  { phrases: ["rogue poison stun", "poison stun"], status: "rogue_poison_stun" },
  {
    phrases: ["rogue poison anti heal", "poison anti heal"],
    status: "rogue_poison_anti_heal",
  },
  {
    phrases: ["hunter trap", "freezing trap"],
    status: "hunter_trap",
    topic: "ultimate_hunter",
  },
  { phrases: ["mage burst", "burst"], status: "mage_burst", topic: "ultimate_mage" },
  {
    phrases: ["priest freedom", "blessing of freedom", "blessing freedom", "freedom"],
    status: "priest_freedom",
  },
  { phrases: ["crippling poison", "rogue poison", "poison"], topic: "ultimate_rogue" },
  { phrases: ["warrior charge", "charge"], topic: "ultimate_warrior" },
  {
    phrases: ["priest salvation", "holy word salvation", "salvation"],
    topic: "ultimate_priest",
  },
];

/** @typedef {{slot?:number, class_id?:number, team_id?:number, role?:string}} Person */

/** Read common researcher questions; unsupported conditions stay explicit.
 * @param {string} query @param {string[]} classNames
 */
export function readMetricQuery(query, classNames) {
  const input = equivalents(tokens(query));
  const giving = input.includes("done");
  const negative = input.find((word) =>
    [
      "no",
      "not",
      "never",
      "without",
      "excluding",
      "exclude",
      "except",
      "after",
      "before",
    ].includes(word),
  );
  if (negative)
    return {
      error: `No matching measurement: this search does not support “${negative}” conditions. Try a measurement name or its exact CSV column.`,
    };
  const receiving = input.some(
    (word) => prefix(word, "received") || word === "receiving",
  );
  const hadCondition =
    input.includes("while") || input.includes("during") || input.includes("under");
  /** @type {string|null} */ let status = null;
  /** @type {string|null} */ let topic = null;
  for (const effect of EFFECTS) {
    if (takePhrase(input, effect.phrases)) {
      status = effect.status ?? null;
      topic = effect.topic ?? null;
      break;
    }
  }
  if (hadCondition && status === null) {
    const effect =
      topic === "ultimate_rogue"
        ? "rogue_poison"
        : topic === "ultimate_warrior"
          ? "warrior_charge"
          : null;
    if (effect && takePhrase(input, ["slowed", "slow"])) status = `${effect}_slow`;
    else if (effect && takePhrase(input, ["stunned", "stun"]))
      status = `${effect}_stun`;
    else if (effect === "rogue_poison" && takePhrase(input, ["anti heal"]))
      status = "rogue_poison_anti_heal";
  }
  if (status) {
    for (const word of [
      "slowed",
      "stunned",
      "active",
      "effect",
      "with",
      "while",
      "during",
      "under",
    ]) {
      const position = input.indexOf(word);
      if (position >= 0) input.splice(position, 1);
    }
  }
  /** @type {string[]} */ const qualifiers = [];
  /** @type {string[]|null} */ let kinds = null;
  for (const measure of MEASURES) {
    if (takePhrase(input, measure.phrases, true)) {
      kinds = measure.kinds;
      if (measure.qualifier) qualifiers.push(measure.qualifier);
      break;
    }
  }
  if (
    status === "hunter_trap" &&
    !kinds &&
    takePhrase(input, ["break", "broken", "period"], true)
  )
    kinds = ["trap_break"];
  const abilityCount =
    topic !== null &&
    !hadCondition &&
    ((input.includes("count") && kinds === null) ||
      Boolean(
        kinds?.some((kind) => ["activation", "status_application"].includes(kind)),
      ));
  const statusCount =
    status !== null &&
    !abilityCount &&
    !hadCondition &&
    kinds === null &&
    input.includes("count");
  const countOnly = (abilityCount || statusCount) && input.includes("count");
  if (statusCount) kinds = ["status_application"];
  else if (abilityCount) kinds = ["activation", "status_application"];
  if (status && kinds?.includes("activation") && !abilityCount)
    kinds = ["status_application"];
  if (status && !kinds && takePhrase(input, ["time", "duration", "step"], true))
    kinds = ["status_time"];
  /** @type {string|null} */ let ability = null;
  for (const value of ["basic", "ultimate"]) {
    if (takePhrase(input, [value])) {
      ability = value;
      break;
    }
  }
  if (
    !ability &&
    !hadCondition &&
    topic &&
    kinds?.some((kind) =>
      [
        "damage",
        "healing",
        "effective_healing",
        "excess_healing",
        "activation",
        "kill",
        "rescue",
      ].includes(kind),
    )
  )
    ability = "ultimate";
  if (
    topic === "ultimate_mage" &&
    kinds?.some((kind) => ["damage", "kill"].includes(kind))
  )
    ability = "burst";
  for (const [phrase, qualifier] of [
    ["accepted", "accepted"],
    ["rejected", "rejected"],
    ["submitted", "submitted"],
    ["allocation", "allocation"],
    ["participation", "participation"],
    ["contribution", "contribution"],
    ["opportunity", "opportunity"],
    ["fraction", "fraction"],
    ["share", "fraction"],
    ["solo", "solo"],
    ["single contributor", "solo"],
    ["multiple contributor", "multi"],
  ]) {
    if (takePhrase(input, [phrase])) qualifiers.push(qualifier);
  }
  const joinedPeople = input.includes("and");
  /** @type {Person[]} */ const people = [];
  /** @type {string|null} */ let marker = null;
  /** @type {Person|null} */ let person = null;
  /** @type {string[]} */ const remaining = [];
  for (let i = 0; i < input.length; i++) {
    const word = input[i];
    if (word === "measure-boundary") {
      person = null;
      continue;
    }
    if (RELATIONS.has(word)) {
      person = null;
      marker =
        word === "from"
          ? "source"
          : word === "by"
            ? kinds?.some((kind) =>
                ["outcome", "score", "return", "episode"].includes(kind),
              )
              ? "subject"
              : receiving
                ? "recipient"
                : "source"
            : ["to", "on", "against"].includes(word)
              ? "recipient"
              : null;
      continue;
    }
    if (FILLER.has(word) || ORIENTATIONS.has(word)) continue;
    /** @type {Person|null} */ let part = null;
    if (word === "team" && ["a", "b"].includes(input[i + 1]))
      part = { team_id: input[++i] === "a" ? 1 : 2 };
    else if (word === "agent") {
      if (input[i + 1] === "id") i++;
      if (/^\d+$/u.test(input[i + 1] ?? "") && !/^[0-9]$/u.test(input[i + 1]))
        return { error: "Agent IDs run from 0 to 9. Use the recorded agent ID." };
      if (/^[0-9]$/u.test(input[i + 1] ?? "")) part = { slot: Number(input[++i]) };
    } else {
      const classId = classNames.findIndex(
        (name, id) =>
          id > 0 &&
          name &&
          (prefix(word, name.toLowerCase()) || word === `${name.toLowerCase()}s`),
      );
      if (classId > 0) part = { class_id: classId };
    }
    if (!part) {
      remaining.push(word);
      continue;
    }
    if (
      person &&
      part.team_id !== undefined &&
      person.team_id !== undefined &&
      part.team_id !== person.team_id
    )
      return {
        error:
          "That identity names both teams. Name one source team and one recipient team, using From and To.",
      };
    if (!person || Object.keys(part).some((key) => Object.hasOwn(person ?? {}, key))) {
      person = { ...part, ...(marker ? { role: marker } : {}) };
      people.push(person);
      marker = null;
    } else Object.assign(person, part);
  }
  if (
    people.length === 3 &&
    Object.keys(people[0]).length === 1 &&
    people[0].class_id !== undefined
  ) {
    const source = people
      .slice(1)
      .find((person) => person.role === "source" && person.class_id === undefined);
    if (source) {
      source.class_id = people[0].class_id;
      people.shift();
    }
  }
  const pair = kinds?.includes("formation") ?? false;
  const patient =
    receiving ||
    qualifiers.includes("opportunity") ||
    (kinds?.length === 1 &&
      [
        "death",
        "status_time",
        "regeneration",
        "poison_prevention",
        "freedom",
        "respawn",
      ].includes(kinds[0]));
  if (!pair) {
    const teamOnly =
      people.length === 1 &&
      people[0].team_id !== undefined &&
      people[0].class_id === undefined &&
      people[0].slot === undefined;
    const namesAbility =
      (topic !== null && !hadCondition) ||
      kinds?.some((kind) => ["activation", "status_application"].includes(kind));
    const defaultRole =
      (kinds || (topic !== null && !hadCondition)) &&
      !(teamOnly && !receiving && !giving && !namesAbility)
        ? patient
          ? "recipient"
          : "source"
        : "subject";
    for (let i = 0; i < people.length; i++) {
      if (!people[i].role) {
        const other = people.find((person, j) => j !== i && person.role);
        people[i].role = other
          ? other.role === "source"
            ? "recipient"
            : "source"
          : defaultRole;
      }
    }
  }
  if (
    (joinedPeople && !pair && people.length > 1) ||
    people.length > 2 ||
    (!pair && people.length === 2 && people[0].role === people[1].role)
  ) {
    return {
      error:
        "The source and recipient are unclear. Try “damage from Warrior to Priest”.",
    };
  }
  if (
    !kinds &&
    !topic &&
    !status &&
    !ability &&
    !qualifiers.length &&
    !people.length &&
    !remaining.length
  )
    return { error: "Enter a measurement name, an agent, or an exact CSV column." };
  return {
    kinds,
    abilityCount,
    countOnly,
    giving,
    receiving,
    status,
    topic,
    ability,
    qualifiers,
    people,
    pair,
    remaining,
    hadCondition,
  };
}

/** @param {Record<string, any>[]} measurements @param {Record<string, any>[]} [topics]
 * @param {Record<string, any>[]} [agents] @param {string[]} [classNames]
 */
export function buildMetricSearchIndex(
  measurements,
  topics = [],
  agents = [],
  classNames = [],
) {
  if (!classNames.length) {
    classNames = [];
    for (const agent of agents) classNames[agent.class_id] = agent.class_name;
  }
  const topicNames = new Map(topics.map((topic) => [topic.name, topic.label]));
  const entries = [...new Map(measurements.map((row) => [row.name, row])).values()]
    .sort((a, b) => a.order - b.order)
    .map((row) => ({
      row,
      facts: row.search_facts,
      name: words(row.name),
      own: tokens([row.label, row.subject, row.unit].join(" ")),
      keywords: tokens(
        [
          row.name,
          row.label,
          row.subject,
          row.unit,
          ...(row.search_terms ?? []),
          ...(row.locations ?? []).map(
            (/** @type {{topic:string}} */ location) =>
              topicNames.get(location.topic) ?? "",
          ),
        ].join(" "),
      ),
      description: tokens(
        [
          row.description,
          ...Object.values(row.topic_text ?? {}).map((text) => text.description ?? ""),
        ].join(" "),
      ),
    }));
  return {
    entries,
    agents,
    classNames,
    exact: new Map(entries.map((entry) => [entry.name, entry.row])),
  };
}

/** @param {Person} person @param {Record<string, any>} identity */
function samePerson(person, identity) {
  return (
    (person.slot === undefined || person.slot === identity.slot) &&
    (person.class_id === undefined || person.class_id === identity.class_id) &&
    (person.team_id === undefined || person.team_id === identity.team_id)
  );
}

/** @param {Record<string, any>} row @param {Record<string, any>} facts
 * @param {Record<string, any>[]} agents @param {Person} person @param {string} role
 */
function matchesPerson(row, facts, agents, person, role) {
  const at = (/** @type {number} */ slot) =>
    agents.find((agent) => agent.slot === slot);
  if (role === "subject")
    return ["team", "team_recipient"].includes(row.scope)
      ? person.slot === undefined &&
          person.class_id === undefined &&
          person.team_id === row.subjects[0]
      : row.subjects.some((/** @type {number} */ slot) => {
          const identity = at(slot);
          return identity && samePerson(person, identity);
        });
  let identity;
  if (row.scope === "source_recipient")
    identity = at(row.subjects[role === "source" ? 0 : 1]);
  else if (row.scope === "team_recipient" && role === "recipient")
    identity = at(row.subjects[1]);
  else if (row.scope === "agent" && facts.subject === role)
    identity = at(row.subjects[0]);
  if (identity) return samePerson(person, identity);
  if (person.slot !== undefined) return false;
  if (role === "recipient") {
    return (
      row.scope === "team" &&
      facts.subject === "recipient" &&
      person.class_id === undefined &&
      (person.team_id === undefined || person.team_id === row.subjects[0])
    );
  }
  if (
    ["none", "unknown"].includes(facts.source) ||
    facts.qualifiers.includes("combined") ||
    facts.qualifiers.includes("opportunity")
  )
    return false;
  if (person.class_id !== undefined && facts.source_class_id !== person.class_id)
    return false;
  let team;
  if (["team", "team_recipient"].includes(row.scope) && facts.subject !== "recipient")
    team = row.subjects[0];
  else {
    const recipientTeam =
      row.scope === "team" ? row.subjects[0] : at(row.subjects.at(-1))?.team_id;
    if (facts.relation === "ally" || facts.relation === "self") team = recipientTeam;
    else if (facts.relation === "enemy" && recipientTeam) team = 3 - recipientTeam;
  }
  return person.team_id === undefined || team === person.team_id;
}

/** @param {ReturnType<typeof buildMetricSearchIndex>} index @param {string} query */
export function searchMeasurements(index, query) {
  const exact = index.exact.get(words(query));
  if (exact) return { matches: [exact], message: null };
  if (!words(query)) return { matches: [], message: null };
  const request = readMetricQuery(query, index.classNames);
  if (request.error) return { matches: [], message: request.error };
  const {
    kinds,
    abilityCount,
    countOnly,
    giving,
    receiving,
    status,
    topic,
    ability,
    qualifiers = [],
    people = [],
    pair,
    remaining = [],
    hadCondition,
  } = request;
  const candidates = index.entries.filter(({ row, facts }) => {
    if (!facts) return false;
    if (countOnly && row.unit !== "count") return false;
    if (kinds && !kinds.includes(facts.kind)) return false;
    if (ability && facts.ability !== ability) return false;
    if (status && hadCondition && facts.status !== status) return false;
    if (status && hadCondition) {
      const conditionSubject =
        status === "mage_burst" &&
        !(receiving && !people.some((person) => person.role === "source"))
          ? "source"
          : "recipient";
      if (facts.status_subject !== conditionSubject) return false;
    }
    if (status && !topic && facts.status !== status) return false;
    if (
      topic &&
      !hadCondition &&
      !row.locations.some(
        (/** @type {{topic:string}} */ location) => location.topic === topic,
      )
    )
      return false;
    if (hadCondition && !status) return false;
    if (!qualifiers.every((qualifier) => facts.qualifiers.includes(qualifier)))
      return false;
    if (pair) {
      if (row.scope !== "ally_pair" && people.length) return false;
      if (people.length === 2) {
        const first = index.agents.find((a) => a.slot === row.subjects[0]);
        const second = index.agents.find((a) => a.slot === row.subjects[1]);
        return (
          first &&
          second &&
          ((samePerson(people[0], first) && samePerson(people[1], second)) ||
            (samePerson(people[1], first) && samePerson(people[0], second)))
        );
      }
      return people.every((person) =>
        row.subjects.some((/** @type {number} */ slot) => {
          const a = index.agents.find((a) => a.slot === slot);
          return a && samePerson(person, a);
        }),
      );
    }
    if (receiving && !people.length && facts.subject !== "recipient") return false;
    if (
      giving &&
      !people.some((person) => person.role === "recipient") &&
      facts.subject === "recipient"
    )
      return false;
    return people.every((person) =>
      matchesPerson(row, facts, index.agents, person, person.role ?? "subject"),
    );
  });
  const covers = (/** @type {string[]} */ field) =>
    remaining.every((term) => field.some((word) => prefix(term, word)));
  const direct = candidates.filter((item) => covers(item.keywords));
  const matches = (
    direct.length ? direct : candidates.filter((item) => covers(item.description))
  )
    .map((item) => ({
      row: item.row,
      direct: Number(covers(item.own)),
      teamApplications:
        (abilityCount || (topic && !kinds)) && item.row.scope === "team"
          ? Number(item.facts.kind === "activation") * 2 +
            Number(item.facts.kind === "status_application")
          : 0,
      namedEffect: Number(
        Boolean(topic) &&
          (item.facts.ability !== null ||
            item.facts.status !== null ||
            item.row.primary_topic === topic),
      ),
      applicable: Number(item.row.applicable),
      receiving: Number(
        receiving && people.length < 2 && item.facts.subject === "recipient",
      ),
      detail:
        Number(!status && item.facts.status !== null) +
        Number(!ability && item.facts.ability !== null) +
        Number(
          !qualifiers.includes("fraction") &&
            item.facts.qualifiers.includes("fraction"),
        ),
    }))
    .sort(
      (a, b) =>
        b.applicable - a.applicable ||
        b.teamApplications - a.teamApplications ||
        b.direct - a.direct ||
        b.namedEffect - a.namedEffect ||
        b.receiving - a.receiving ||
        a.detail - b.detail ||
        a.row.order - b.row.order,
    )
    .map((item) => item.row);
  return {
    matches,
    message: matches.length
      ? null
      : "No measurement matches that request. Try a measurement name or its exact CSV column.",
  };
}

/** @param {ReturnType<typeof buildMetricSearchIndex>} index @param {string} query */
export function findMeasurements(index, query) {
  return searchMeasurements(index, query).matches;
}
