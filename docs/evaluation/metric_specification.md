# Evaluation Metric Specification

## Status and authority

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** Metric rows remain subject to
> their stated readiness and validity gates.

This document is the normative MARL-BattleGrounds metric contract. It owns
metric identities, meanings, dispositions, amount stages, opportunity rules,
attribution limits, and presentation tiers. The companion
[evaluation protocol](protocol.md) owns evaluation cells, aggregation,
uncertainty, cross-play, scenarios, and leakage control. Accepted departures
from the original PDF are recorded in the
[specification amendments](../design/specification_amendments.md).

The specification is deliberately broader than the initial implementation.
`derivable_now` means the completed mechanics and Milestone 6 evaluation seam
can support the metric. `requires_future_task_authority` means the disposition
is settled but the formula cannot activate until the owning task defines its
score, objective, reward, and terminal facts. A deferred or blocked row is not
an official result.

Stable IDs are public semantic references for documentation, artifacts and
tests. Formulas may be implemented only by their named owner.

## Current scalar TDM contract

### Manuscript family summary

The table below groups measurements by what they tell us. A group can have many
columns because it records numbers for different teams, agents, targets, or
statuses. For example, damage from Agent 0 to Agent 5 and damage from Agent 0 to
Agent 6 need separate columns. They describe the same kind of measurement.
Each row of a run table describes one episode. Each numerical column is counted
once here, even if several viewer groups show it. The data dictionary also lists
the columns that identify the run, episode, and agents.

<!-- metric-family-summary:start -->
| Measurement family | Numerical columns | What the family describes |
| --- | ---: | --- |
| Episode results | 26 | Episode length, outcomes, returns, scores, kills, and deaths. |
| Ability use and action acceptance | 754 | How often abilities were used, who they targeted, and which actions the game rejected. |
| Deaths and respawning | 48 | Who died, how long agents were dead, and when they returned. |
| Kills and coordination | 746 | Who helped kill each enemy, who killed alone, and whether attackers chose the same target. |
| Damage | 538 | Who dealt damage, who took it, which ability dealt it, and each agent's share. |
| Healing and excess | 2,028 | Who healed whom, which ability was used, how much healing was useful or excess, and automatic health recovery. |
| Effects on controlled recipients | 3,592 | Damage, healing, and kill credit when the affected agent already had a named harmful status. |
| Status applications, duration, and Freedom | 1,532 | Who applied each status, who had it and for how long, and when Freedom protected movement. |
| Trap breaks | 218 | Who was trapped, who broke each Trap with damage, and how much Trap time remained. |
| Burst | 520 | Damage dealt during Burst, who took it, and who helped with kills while Burst was active. |
| Auras | 556 | Who gave and received aura coverage, extra damage from Mage auras, and damage blocked by Warrior auras. |
| Poison healing prevention | 12 | Priest healing prevented on affected agents and teams. |
| Lethal-damage rescues | 544 | Who could be saved from lethal damage, who survived, and which healers and abilities helped. |
| Formation | 44 | Distances between living teammates and how many times each pair was measured. |
| **Total** | **11,158** | **Unique numerical columns; shared viewer appearances are counted once.** |
<!-- metric-family-summary:end -->

Allocation asks where an agent's output went. Contribution asks how much that
agent supplied compared with its team. Suppose Agent 3 deals 100 damage, with
40 going to Agent 8. Team A deals 80 damage to Agent 8, including Agent 3's 40.
Then 40% of Agent 3's damage went to Agent 8, and Agent 3 supplied 50% of Team A's
damage to Agent 8.

Keep the named ability, effect and time rule on both sides of each fraction.
For example, `agent_5_to_agent_3_burst_damage_allocation_fraction` divides
Agent 5's Burst-active damage to Agent 3 by all Agent 5's damage while Burst
was active. A value of `0.4336` means 43.36% went to Agent 3. It does not say
Agent 5 supplied 43.36% of the damage Agent 3 received. Damage or healing to
agents with a named harmful effect checks each recipient at the start of the
tick; an effect first applied later that tick does not count.

Named effects also retain distinct meanings when their counts coincide: two
Charges produce two Charge activations, two Charge Slow applications, and two Charge
Stun applications. These are descriptions of the same two casts, not six casts.
Several teammates can get credit for the same kill. If two agents both helped
with the only kill, each has 100% participation. Adding these gives 200%; it
does not mean there were two kills.
If two Mages deal damage while Burst is active and kill one enemy together,
that is two Burst kill contributions but one enemy killed with Burst help.
Only damage and useful Priest healing on that enemy's death tick earn kill
credit. Earlier damage does not turn a later one-helper kill into a shared kill.

A blank CSV cell means the value does not apply or cannot be calculated. For
example, an inactive agent has no measurements. An agent that made no casts has
no share of casts on a target: there is no total to divide by. An active agent
that could deal damage but dealt none has a real zero.

Averaging episode fractions gives every episode the same weight. Dividing the
total count by the total number of chances gives each chance the same weight.
For example, 1/1 and 1/9 average to about 55.6%, but 2/10 is 20%. Choose the
question you want to answer; the tables provide the counts for both.

### Schema and compatibility

The current schema is `marlbg.tdm.scalar@13`: **26 priority numeric columns** and
**11,158 full numeric columns**, with priority included in full. The viewer has
27 topics and 43 tables; different topics can show the same exported column. Column order and
names remain identical across valid one-through-five-agent, asymmetric,
permuted-class and repeated-class rosters. This schema supersedes the historical
46-ID report catalog for new computation. It contains no KOTH or CTF metrics.
The former host `full=True` reducer factory and unused draft accumulator are
retired. `build_tdm_metric_reducers()` retains only the five basic V1 outcome
statistics needed to reproduce archived samples; new full computation uses
`evaluate(..., metrics="full")` or the corresponding environment mode.
Historical V1 component/report schemas and readers remain supported. Scalar
schema 3 preserved all 2,048 schema-2 measurements and added 9,272 columns.
Schema 4 changed relationship prefixes on 9,400 existing columns without adding
or removing measurements, changing their meanings, or changing numerical order.
Schema 5 removed 206 columns and put related full measurements next to each
other. Surviving names and calculations stayed the same. It removed 158 values
that are always zero or one when defined, 40 repeated self counts, and eight
class totals that can be rebuilt from agent rows. The removal list does not
depend on which classes appear in a sampled game. A class can move between
slots, appear several times, or be absent; the remaining headers stay fixed.
Schema 6 added 12 Basic kill fractions. It also placed the existing single- and
multiple-contributor kill counts and fractions in both **Kill Contributions**
and **Team Coordination**, using the same CSV columns in both groups. Schema 7
adds 20 columns in total: a Basic kill count and share of all team kills for
each class on each team. Schema 8 kept every measurement and changed full-column
order to follow each column's primary topic and view. Schema 9 changed the healing
term to **Excess**, including its CSV names and topic name. It kept all 11,146
measurements in the same order, with the same values and rules for blank cells.
Schema 10 removes 46 observed-respawn-wait columns. It keeps team wave counts,
mean agents returned per wave, and mean observed waiting ticks under
**Respawning**. Death counts and time dead remain in **Deaths and Time Dead**.
Schema 11 added 40 recipient columns from existing counters: each agent's Trap
period count, Trap-break fraction, mean Trap time left at a break, and chances
for its team to save it with Priest healing. Existing names, values and relative
order stay the same. An agent's rescue opportunity describes that agent as the
patient; it does not claim that the agent is a Priest or can save someone else.
Schema 12 added 12 Basic healing-save shares using existing counts. It kept all
11,140 earlier column names and values, clarified names and explanations, and
put related healing amounts and fractions together. Schema 13 adds six team
ability application columns: two each for Charge, Poison and Salvation. All
existing status-column names and meanings stay unchanged. The five
named Ultimate topics may use their ability's name in row labels and tooltips;
other topics keep the shared column names. A wording change or another appearance
in a topic does not add a measurement. Neither addition needs new per-tick counters.
The full CSV is ordered by primary topic, then view, team, agent, recipient,
and source-to-recipient detail; priority retains its original 26-value order.
Within each subject, All abilities come before Basic and then Ultimate. Each
amount is followed by its fractions, with allocation before contribution.
The Viewer uses the same order and the same exported measurements.
Existing CSV files are not rewritten. Incompatible schema resumption fails before
recovery, truncation, or writing; start a new run when the schema changes.

### Find a measurement by topic or CSV name

Choose a **Topic**, then **Totals** or **By Recipient** when the topic has both.
Totals show team-wide and acting-agent measurements. By Recipient shows totals
for affected agents, then team-to-agent and agent-to-agent details. The two
tables for one topic share no rows. Team Formation and ten other focused topics
need only one table. Six headings keep the Topic dropdown easy to scan.

Use **Find a Measurement** to search a plain name or an exact CSV column. Search
covers all 11,158 numerical measurements, even ones that do not apply to this
replay. An inapplicable result explains why and keeps its definition and CSV
name available. It does not insert an impossible row into the table. A zero
denominator is different: the measurement applies, but its fraction is blank.
The search list has no tick values and is loaded once per replay analysis.

The tables use the recorded classes and active slots to hide impossible roles.
For example, a Priest can help save a Warrior: the Priest appears as the healer,
and the Warrior appears as the saved ally. A Warrior cannot appear as a healing
helper. Moving the Priest to another slot moves these rows with it.

The same rule covers all topics. Solo kills and damage-caused Trap breaks need
a damaging agent. Priest kill support needs a damaging teammate. Burst damage
received needs an opposing Mage; aura coverage received needs a matching aura
giver on the same team. Priest healing prevented by Poison needs a friendly
Priest, but not a current Rogue: Poison may already be active when recording
starts. Initial Trap, stun and Freedom rows also remain without their original
caster. No filter uses the current score, observed zero values, range, cooldowns
or whether an agent is currently alive. Healing Done shows Priests as healers;
Damage Done shows agents whose recorded abilities can deal that damage. A
capable agent keeps its zero amounts and blank fractions before it acts.
Other classes still appear as patients when a friendly Priest can heal them.
Regeneration and combined healing received remain visible without a Priest.
An already-stunned Priest needs another Priest to heal it.

Focus fire needs two damaging teammates. A shared kill needs two attackers,
or an attacker and a Priest who helps through healing. Team Formation needs
two active teammates. Match results, deaths, time dead and
initial status records remain available even when no current agent can cause
another such event.

These are display rules. All CSV columns and their values remain available,
and search explains why a measurement does not apply to the recorded roster.
Rescue opportunities still use actual legal healing range at the start of the
tick. They do not use personal observation range or a separate fixed radius.

Exact CSV names come first. Otherwise, the measurement's own name and recorded
agent come before mentions in related topics. Ordinary singular/plural words
both work: Warrior death, Warrior deaths, and Warrior death counts find the
Warrior's death count. You can also type the start of a word, such as regen.

Each column has one fixed primary location in the CSV. Related topics can reuse
it. For example, Kill Contributions owns the shared-kill counts that Team
Coordination also shows. Excess Healing owns excess amounts and fractions, while
the general healing tables reuse those measurements beside useful healing.
Shared columns keep the same display name outside the five named Ultimate
topics. Those five topics use Mage Burst, Warrior Charge, Hunter Trap, Rogue
Poison and Priest Salvation in their row and tooltip wording. This changes no
value or CSV name. Useful shared death and save totals
name the all-ability denominator of the displayed participation fractions.
Burst has no displayed kill fraction using Total Kills, so its table omits that
shared row. Total Kills remains in Episode Results and in the CSV.
Generic Ultimate damage stays under Damage Done when a slot changes class.
An Ultimate table may use only columns whose primary locations are elsewhere;
a zero in the table below means it adds no separate columns, not that its
measurements have zero values.

The column guide records `primary_topic`, `primary_view`, `gui_groups`,
`full_run_column_number`, and `replay_column_number`. Column numbers start at
one and include each export's own identity fields. These guide fields do not
add measurements to the episode tables. The scientific family table above
counts concepts separately from these navigation locations.

<!-- metric-navigation:start -->
| Topic | Primary columns in Totals | Primary columns By Recipient | Primary columns in single view |
| --- | ---: | ---: | ---: |
| Episode Results | — | — | 26 |
| Ability Activations | 24 | 640 | — |
| Accepted and Rejected Actions | — | — | 84 |
| Damage Done | 46 | 450 | — |
| Damage Received | — | — | 42 |
| Damage to Enemies With Harmful Effects | 84 | 1120 | — |
| Healing Done | 118 | 1050 | — |
| Healing Received | — | — | 148 |
| Excess Healing | 72 | 640 | — |
| Healing to Allies With Harmful Effects | 84 | 1030 | — |
| Priest Healing Saves | 74 | 470 | — |
| Kill Contributions | 142 | 600 | — |
| Kills of Enemies With Harmful Effects | 154 | 1120 | — |
| Deaths and Time Dead | — | — | 42 |
| Respawning | — | — | 6 |
| Team Coordination | — | — | 4 |
| Team Formation | — | — | 44 |
| Aura Coverage | 72 | 480 | — |
| Damage Added or Blocked by Auras | — | — | 4 |
| Status Applications | 108 | 1280 | — |
| Time With Status Effects | — | — | 108 |
| Freedom Against Slows | — | — | 36 |
| Burst (Mage Ultimate) | 50 | 470 | — |
| Charge (Warrior Ultimate) | 2 | 0 | — |
| Freezing Trap (Hunter Ultimate) | 28 | 190 | — |
| Crippling Poison (Rogue Ultimate) | 4 | 10 | — |
| Holy Word: Salvation (Priest Ultimate) | 2 | 0 | — |
| **Total** | | | **11,158 unique measurements** |
<!-- metric-navigation:end -->

Schema 8 preserved all schema-7 names, definitions, calculations and blank-value
rules. Schema 9 changed the excess-healing names only. Schema 10 removed 46 wait
measurements, schema 11 added 40 recipient measurements, schema 12 added 12
Basic healing-save shares, and schema 13 adds six team ability application
counts. The full-run table has 30 identity fields plus 11,158 measurements:
11,188 columns in total. Replay exports keep their 49 identity
fields, giving 11,207 columns. Historical files stay untouched;
the writer rejects older schemas before recovery or writing. Use names rather
than old column positions when comparing exports across these versions.

### Schema 13 team ability application columns

Schema 13 adds six columns that name the whole ability. It renames no existing
column. Slow, Stun, Anti-Heal and other status-application columns keep their
exact names, meanings and Status Applications rows.

| Added CSV columns | Count | Ultimate topic |
| --- | ---: | --- |
| `team_{a,b}_warrior_charge_applications` | 2 | Charge |
| `team_{a,b}_rogue_poison_applications` | 2 | Crippling Poison |
| `team_{a,b}_priest_holy_word_salvation_applications` | 2 | Holy Word: Salvation |

Here `{a,b}` means one column for Team A and one for Team B. Each new column
appears only in its named Ultimate topic and counts that team's activations of
that ability. An allowed activation counts once, including a repeat use or
Salvation healing that is entirely excess. A rejected action does not count.
Several matching agents' activations are added together. A team with an active
agent of that class and no activations has a real zero; a team with no active
agent of that class has a blank value. Mage Burst and Hunter Trap keep their
existing team ability application columns.

These columns reuse existing activation counts when the full result is built.
They add no per-tick counters or counter updates. The six values and their
valid/blank flags add 30 logical output bytes per environment, bringing the full
result to 55,790 bytes. This is a size calculation, not a speed or peak-memory
measurement. The 11,152 earlier measurements keep their exact names, meanings
and relative order. Priority remains 26 values.
Existing CSV files stay untouched;
an older schema cannot resume under schema 13. The writer rejects that mismatch
before recovery, truncation or writing. The replay file format is unchanged.

The five-topic audit separates numerical equality from correct column names.
Charge and Poison ability totals must point to their own ability columns, even
when their counts equal the existing Slow application counts. Those status
columns remain useful separate measurements. Priest also needed team Salvation
counts. A named-ability count search could separately select the wrong kind of
row. Checking equal values alone did not prove that the right CSV column appeared
in the ability table; the check must also inspect its exact name and topic.
The schema-13 check passed 157 name, order, roster and CSV-position checks across
five roster layouts. Each team's first row in all five Ultimate tables names
the matching ability column when that class is present. It also checked all
11,152 earlier definitions and the retained team status rows. The raw checks
are in `artifacts/m8-search-direction/ultimate-team-column-audit.json`.
These checks establish metadata mapping, not replay values or browser behavior.

Ability wording must preserve each fraction's full meaning. A source's share
of team applications still divides by the team's combined applications from
all five abilities. A source's share of team direct ability damage still
includes Warrior Charge, Hunter Trap and Rogue Poison in the team total.
Kill and healing-save participation keep all kills or all saves as their
denominators. No tooltip may turn these into a one-ability total.

### Schema 12 Basic healing-save shares

Schema 12 added the matching Basic shares beside the existing Basic save counts.
Here `i` is an agent slot from 0 to 9. Slots 0–4 belong to Team A; slots 5–9
belong to Team B. `{a,b}` names the source team.

| Added columns | Count | Meaning |
| --- | ---: | --- |
| `agent_i_basic_rescue_participation` | 10 | Numerator: Saves this Priest helped with its Basic healing. Denominator: All unique saves by its team, from any ability. |
| `team_{a,b}_basic_rescue_fraction` | 2 | Numerator: Saves helped by at least one Basic heal from this team, counting each saved ally once per tick. Denominator: All unique saves by this team, from any ability. |

A save means the ally survives a tick whose incoming damage alone would have
killed it. A contributing Priest must supply useful healing on that tick.
The source counts already record who helped. The team counts already count
each saved ally once per tick. These fractions reuse those counts when the full
result is built; the per-tick counters and their updates stay unchanged.

If two Priests use Basic healing to help with the team's only save, each agent's
Basic participation is `1.0`, and the team's Basic save fraction is `1.0`.
There are two helpers and one save. If Basic and Ultimate healing both help
with that save, both team ability fractions are `1.0`. The fractions can
overlap; adding them does not give the number of saves.

These fractions are blank when the team has no saves. Inactive agents' shares
are also blank. The viewer shows healer rows only for Priests in active slots;
moving or repeating Priests does not change the fixed CSV columns. These are
shares of actual saves, not shares of rescue opportunities or proof that one
Priest could save the ally alone. Existing recipient-specific Basic shares
keep their own denominators: all unique saves of that recipient.

The twelve values and their valid/blank flags add 60 logical bytes to each full
result. Schema 12 had 55,760 logical output bytes per environment. This is a
size calculation, not a speed or peak-memory measurement. That migration needed
a new run: a schema-11 run could not resume under the schema-12 header.

### Schema 10 respawn cleanup

Respawning has three columns per team: `respawn_waves`,
`mean_agents_per_respawn_wave`, and `mean_observed_respawn_wait_steps`. Names
start with `team_a_` or `team_b_`, giving six columns altogether. Waves count
scheduled respawn times, including times when nobody returns. Mean agents per
wave divides all returning agents by all those waves, including empty ones.
For example, a wave with no returning agents counts as one wave with a mean
of zero. Before the first wave, the mean is blank. Mean waiting ticks is blank
when no waiting period was seen.

Mean waiting ticks uses the dead time seen in this recording, divided by the
number of waiting periods seen. It includes waits already underway at the start
and waits still open at the selected tick. For example, two waits seen for
two and four ticks give a mean of three ticks, even if one is still open. Each
wait counts equally; this is not an average of individual agents' averages.

Schema 10 removes all team and agent versions of `observed_respawn_waits`,
`unfinished_respawn_waits`, and `unknown_start_respawn_waits`, plus the ten
agent versions of `mean_observed_respawn_wait_steps`. That is 46 removed columns.
The two team means stay. No individual-agent wait rows remain in this topic.
This does not change death counts, time dead, respawn events or replay content.

### Schema 7 class Basic kills and migration

Schema 7 adds ten class Basic kill counts and ten fractions: one count and one
fraction for each of the five classes on each team. In these names, `{a,b}` is
`a` for Team A or `b` for Team B. `{class}` is `mage`, `warrior`, `hunter`,
`rogue`, or `priest`.

| Added columns | Count | Meaning |
| --- | ---: | --- |
| `team_{a,b}_{class}_basic_kills` | 10 | Kills helped by at least one Basic ability from this class on the named team. Each enemy death counts once. |
| `team_{a,b}_{class}_basic_kill_fraction` | 10 | Numerator: Kills helped by this class's Basic abilities on the named team, counting each kill once. Denominator: All kills by the named team. |

Credit needs Basic damage on the tick the enemy dies, or useful Priest Basic
healing of a teammate who damaged that enemy on that tick. Healing that is all excess gives no credit. A Mage's Basic attack during Burst still gives Basic
credit.

Repeated classes do not multiply the count. If two Mages help with Basic attacks
on one kill, their team's Mage Basic kill count rises by one. If a Mage and a
Warrior each help with Basic attacks on that kill, both class counts rise by one.
There is still only one kill. If it is the team's only kill, both class fractions
are `1.0`. Class fractions can overlap with each other and with Ultimate help;
adding these fractions does not give the number of kills.

A class present in an active slot has a valid zero count if it has no Basic
kill credit. Its fraction is zero if the team has kills, and blank if the team
has no kills. If the team has no active agent of that class, both its count and
fraction are unavailable.
The denominator always includes all kills by the named team, regardless of
which class or ability helped.

The new running count uses a fixed `(5, 2)` array of 32-bit integers: one count
per class and team, adding 40 logical bytes per full-enabled environment. The
20 output values and their valid/blank flags add 100 logical bytes per full
result. The schema-7 full result needed 55,730 logical bytes. These are storage
calculations; they do not establish a change in speed or measured peak memory.

Existing schema-1/2/3/4/5/6 files remain unchanged. A schema-7 writer rejects an
older scalar schema before recovery, truncation, or writing. Start a new run
directory to use the new columns; the replay file format is unchanged.

### Schema 6 additions and migration

This earlier schema added ten agent fractions and two team fractions, bringing
its full output to 11,126 numerical columns and 11,156 run-table columns with
identity. These measurements remain in schema 7. Here `i` is an agent slot from
0 to 9. Slots 0–4 belong to Team A; slots 5–9 belong to Team B.

| Added columns | Count | Meaning |
| --- | ---: | --- |
| `agent_i_basic_kill_participation` | 10 | Numerator: Kills this agent helped with its Basic ability. Denominator: All kills by its team. |
| `team_a_basic_kill_fraction`, `team_b_basic_kill_fraction` | 2 | Numerator: Kills helped by at least one Basic ability on that team, counting each kill once. Denominator: All kills by that team. |

Basic help means Basic damage on the tick the enemy dies, or useful Priest Basic
healing of a teammate who damaged that enemy on that tick. Healing that is all excess gives no credit.
A Mage's Basic attack while Burst is active still counts as Basic help.
Several agents can earn credit for one kill, but the team's Basic count includes
that kill only once. For example, if two agents help with Basic abilities on the
team's only kill, each agent's Basic participation is `1.0` and the team's Basic
kill fraction is also `1.0`.

A kill can have both Basic and Ultimate help. If a Basic attack and an Ultimate
help with the team's only kill, both team fractions are `1.0`. Adding them gives
`2.0`, which does not mean there were two kills. These fractions are blank when
the source team has no kills; an inactive agent's fraction is also blank.

The existing single- and multiple-contributor kill counts and fractions appear
in both **Kill Contributions** and **Team Coordination**. The groups share each
measurement's one CSV column, so this adds no duplicate output columns.

Schema 6 left existing schema-1/2/3/4/5 files unchanged and rejected an older
scalar schema before recovery, truncation, or writing. Its additions did not
change the replay file format. New runs now use schema 13, as described above.

### Schema 5 removals

This historical change followed the game rules, not a list of zeros from sampled
games. Here `i` is any agent slot, and its matching team is A for slots 0–4 or B
for slots 5–9. All surviving numerical names and meanings remain unchanged.

| Removed measurements | Columns | What remains |
| --- | ---: | --- |
| Class-specific team Ultimate damage | 10 | Individual Ultimate damage and overall team Ultimate damage |
| Mage team kills/fraction from activating Burst | 4 | Damage and kill credit while Burst is active |
| Priest team Ultimate activations | 2 | Individual and overall team activations |
| Self-healing while already under Charge Stun, Trap, or Poison Stun, including allocation/contribution | 90 | Other-source healing of stunned allies and general self-healing |
| Burst self application allocation/contribution fractions | 20 | Named Burst activation counts |
| Mage/Warrior self aura coverage/contribution fractions | 40 | Self covered time and changing self allocation fractions |
| Burst self-pair and team-to-self application counts | 20 | The identical `agent_i_mage_burst_applications` count |
| Mage/Warrior self aura eligible time | 20 | The identical self `_aura_covered_steps` count |
| **Total** | **206** | |

Burst targets its caster, and an eligible aura giver always covers itself.
Their removed fractions are mathematically one whenever defined. The removed
self-healing-under-stun amounts are zero: an already-stunned Priest cannot
activate healing. A different Priest can still heal that stunned ally.

The 516 repeated named-effect columns for Charge and Poison stay. They let a
researcher select an effect by name. This exception concerns applications only;
status durations and effects on controlled agents can differ.

Unique class kills also stay. If two Warriors help kill one enemy, that is two
individual contributions but one class kill. A Priest can earn kill credit by
healing an attacker on the death tick, when some of that healing was effective.
Class changes, repeated classes, inactive slots and missing classes do not
justify dropping generic slot columns such as `agent_4_kill_contributions`.

**Kills Helped by an Ultimate Activated That Tick** counts help from an Ultimate
activated on the tick the enemy dies. That help can be damage, or Priest healing
of an attacker when some of that healing is effective. A Mage's later Basic attack
while Burst is active counts as Basic and Burst help, not as that Mage's Ultimate
contribution. For example, a Mage killing alone with a Basic attack during Burst
adds one Basic-assisted kill and one Burst-assisted kill, with no Ultimate-assisted
kill. If a Warrior's Charge also helps with that kill, the Ultimate count rises
by one too. There is still only one enemy death. These categories overlap; adding
them does not give total kills. Credit does not prove the ability was needed to
get the kill. This wording clarification leaves the CSV names and calculations
unchanged.

### Reading source and recipient names

Read a name from left to right. Before `to` is the agent or team doing something
(the source). After `to` is the agent it affects (the recipient). The rest names
the measurement. Two teammates use `and` when neither acts on the other, such
as their distance apart. The lower agent number comes first.

| Schema 3 prefix | Schema 4 prefix | Relationship | Columns renamed |
| --- | --- | --- | ---: |
| `team_b_agent_1_` | `team_b_to_agent_1_` | Team B is the source; Agent 1 is the recipient. | 130 |
| `agent_5_agent_1_` | `agent_5_to_agent_1_` | Agent 5 is the source; Agent 1 is the recipient. | 9,230 |
| `agent_0_agent_1_` | `agent_0_and_agent_1_` | Unordered teammate pair, such as formation distance. | 40 |

The examples illustrate each prefix rule across its entire scope. The other
1,920 names, including all 26 priority names, remain unchanged. Old CSV files
retain their original headers. The viewer's **CSV Column** tooltip identifies
the installed schema's exact column; no second `from` alias is exported.

Activating an ability and receiving its effect answer different questions.
Agent 0 can **activate its Basic ability 87 times** while **Team A targets
Agent 0 with Basic abilities 49 times**. `agent_0_basic_activations` counts the
former; `team_a_to_agent_0_basic_applications` counts the latter. Each activation
counts once. An attempt the game rejects does not count. Refreshing a status
does not add another activation.

If an agent uses its ability ten times and aims four uses at one target, its
share on that target is **4/10 = 0.4, or 40%**. If the team aimed eight uses at
that target, the agent supplied **4/8 = 0.5, or 50%** of them.

Hover over a measurement, or use the keyboard to focus it, to read its meaning.
**From** and **To** name the source and target. **Numerator** names the amount
being compared; **Denominator** names the total it is compared with. For example,
four Basic ability activations by Agent 3 on Agent 8 out of eight by Team A on
Agent 8 give `0.5`, or `50%`. The help names Team A or Team B explicitly.
**Blank When** explains when no number
can be given. **CSV Column** gives the exact column to use in pandas. Fractions
remain decimals in the viewer and CSV: `0.4` means `40%`.

Every tooltip starts with a short definition for that row. For example:
"How many times this agent used its Basic ability." Counting rules and brief
examples follow where needed. The same wording appears in the generated column
dictionary. Schema 9 uses the term Excess in names and explanations; the
calculations stay unchanged.

**How to Read It** judges the team named in the row. For example, more damage
dealt is better for the attacking team; less damage received is better for the
team taking it. Team B rows do not switch to Team A's point of view. The score
difference names both sides: higher favors Team A and lower favors Team B.

**Context dependent** means the number alone cannot tell us whether the team
played better. A Warrior may take hits to protect teammates, so all its incoming
damage amounts use this label. The viewer reads the class from the replay;
it never assumes a particular slot is a Warrior. Whole-team damage received
still uses "Lower is better."

Damage to an already trapped enemy is context dependent: it hurts the enemy but
can also free it from Trap. The same tradeoff applies to damage received while
trapped, for every class. Other harmful-effect damage keeps its usual guidance.

Excess healing is the part of delivered healing that does not fit under the
recipient's maximum health, after damage on that tick. It is not a judgment
about the action. Basic healing can still give Freedom when all its healing
is excess. Basic and combined excess amounts and efficiency fractions are
therefore context dependent.
Ultimate-only excess amounts and efficiency fractions use "Lower is better":
Holy Word: Salvation does not give Freedom. Shares showing where excess went
remain context dependent.

Effective Priest healing is delivered healing minus that health-cap excess.
It can offset incoming damage even if health does not rise or the ally still
dies. For example, an ally starting with 10 health who takes 30 damage and
receives 5 healing still dies, but all 5 healing is effective. This amount
does not count a successful save; the separate healing-save measurements do.

Keep healing amounts separate from their fractions:

| Measurement | How to Read It |
| --- | --- |
| Effective Priest healing amount, excluding regeneration | Higher is better |
| Total Effective Healing Received, including regeneration | Context dependent |
| Basic/combined effective-healing fraction | Context dependent |
| Basic/combined excess-healing fraction | Context dependent |
| Ultimate-only effective-healing fraction | Higher is better |
| Ultimate-only excess-healing fraction | Lower is better |

Example: a Priest gives 50 effective healing, then heals a full-health ally to
give Freedom. The effective amount stays 50, but its fraction falls. That lower
fraction does not mean the Freedom cast was bad. Each effective/excess pair
sums to one when its shared delivered-healing denominator is nonzero. Both
fractions are blank when that denominator is zero. Allocation and contribution
shares remain contextual: they show where healing went, not how much was useful.

Regenerated healing is context dependent. More can mean leaving a fight at a
good time or avoiding fights too much. Total Effective Healing Received combines
effective Priest healing with regeneration, so it is also context dependent.
Without Priest healing, that total equals Regenerated Healing. The separate
effective Priest-healing amounts keep their existing guidance.
Fully accepted action counts and the acceptance rate
use "Higher is better."

Use these labels to compare similar situations, not to rank policies by every
column. A contribution count shows how many kills or saves an agent
helped with. Its participation fraction shows its share of those events. If a
teammate gets another kill, that fraction can fall even though the team did
better. Participation and target-allocation fractions are context dependent.
Aura coverage and activation frequency are also context dependent: more coverage
can limit positioning, and more casts can give only excess healing.

[The column data dictionary](metric_columns.csv) lists all **11,188 full-table
columns**: 30 identity columns followed by 11,158 numerical measurements. Priority
tables have **56 columns**. The dictionary includes meaning, units, scope,
subjects, subject/recipient roles, numerators, denominators, missingness, priority/full
membership, GUI labels, relevant views, family and defensible direction.
`gui_text_by_topic` contains the row and tooltip wording used in named Ultimate
tables. Only those five topics may replace a label or the words explaining a
numerator, denominator, guidance or blank value. Other topics may add only a
description or subtitle. In the five Ultimate topics, an explicitly empty
subtitle hides an old status caption on an ability row. Other fields and topics
still require nonempty text. This context never changes the numerical value,
denominator or CSV name. The viewer builds it once from the recorded classes.
It uses the writer's identity order and `evaluation/metric_catalog.py`;
`python -m scripts.dev.export_metric_dictionary --check` detects stale exports.
In its `subjects` field, team scope uses Core IDs 1/2, agent scope uses slots 0–9,
team-to-recipient scope uses a Core team ID followed by the recipient's global
slot, source/recipient scope orders source before recipient, and ally pairs are
unordered pairs written with the lower slot first. Class IDs remain recorded row
metadata. The same exporter generates and checks the manuscript family counts.

### Selection and cost boundary

`make()` and `evaluate()` accept one mode: `metrics="none"`, `"priority"` (default),
or `"full"`. `full_metrics_episodes` independently upgrades chosen one-based
planned episode IDs. `replay_episodes` independently requests capture; neither
selection implies the other. Finite integer iterables include `range(...)` and
lists. Resolve selections before compiled transitions. Metric computation needs
authoritative transition facts, not replay files.

Priority records episode length; both teams' win/draw/loss indicators; canonical
agent/team returns; Team A/B scores and A-minus-B difference; team kills/deaths.
Team return de-broadcasts Core's per-agent team reward, so adding active teammates
does not multiply it. Kills count deaths during this episode, independently of
initialized scenario scores. Full includes every retained family below. None
retains only intrinsic environment completion/outcome needed for execution.

Numerical accumulation/finalization is JAX, with fixed-size integer/float32/boolean
arrays and no mandatory trajectory. Floating GPU matrix reductions explicitly
preserve float32 precision. Formatting, host metadata and CSV are outside the
compiled numerical path. The performance command measures these stages separately;
a fast final formatting operation is not evidence for fast metric computation.

### Tables and identity

Each CSV row describes one episode. `priority_metrics.csv` includes priority/full
selected episodes; `full_metrics.csv` includes full-selected episodes and copies
already computed priority values. Tournament `match_results.csv` replaces a
redundant priority file, preserving mandatory outcomes when optional metrics are
disabled. Shared definitions/configurations/provenance live in `run_details.json`.
Without an output destination, return in-memory columns and create no files.

Identity columns are `run_id`, `phase`, `pass_id`, `episode_id`, `seed_id`, `map_id`,
`config_id`, `team_a_policy`, `team_b_policy`, `checkpoint_id`, then
`agent_0_class_id`, `agent_0_active`, through slot 9. Episode IDs are positive
integers assigned before execution; seeds identify random streams independently
of completion order. `map_id` can be missing for a custom configuration; `config_id`
identifies its recorded configuration. Unknown checkpoint/seed information stays
missing. Active flags are 0/1; fixed slot/team mapping is 0–4 Team A, 5–9 Team B.
Unknown training history is never filled with fabricated zero steps or run names.
A training episode spanning parameter updates belongs to an evolving policy.

Unavailable measurements are empty CSV cells; real zeros remain zero. In CSV,
an active Priest's Damage Done and an active Mage's Healing Done are valid zeros.
The Viewer hides those impossible-output rows but search still finds them. Inactive
subjects, absent class-specific capabilities and zero-denominator rates/means are
unavailable. Failures raise errors, never become empty cells. Agent identities use
fixed slots, making curriculum comparisons directly joinable; class names qualify
class-specific effects and team summaries rather than replacing agent identity.
Shared context in an Ultimate view requires the relevant class on the acting
team: a healed recipient uses its own team, while a killed enemy's denominator
uses the opposing team. Inactive slots cannot supply that class. Zero output
remains visible when the recorded agent can produce it. Initial status and Trap
records remain without a current caster. Poison prevention needs a friendly
Priest, but not an opposing Rogue.

`RunWriter.write(infos)` consumes all completion records and selected packets in
a step or rollout chunk. `flush()` acknowledges durability; closing flushes.
Writes append buffered batches. New output destinations get unique child runs;
resumption requires `resume_from`. Recovery uses durable boundaries, preserving
exactly-once completion. Disk failures are also recorded when possible.

### Retained families and attribution

- Ability activations: Basic/Ultimate by agent/team; referenced by other GUI groups.
- Deaths: agent/team counts, fractions, dead-agent steps and fractions of team dead time.
- Kill contributions: direct damage on the authoritative lethal transition plus
  useful same-tick Priest healing of a direct contributor. Deduplicate each
  Priest/enemy/transition; no recursive support chains or credit for healing that is all excess.
  Agent Basic participation divides that agent's Basic credit by all team kills.
  Team Basic fractions count each Basic-helped kill once and use all team kills
  as the denominator. Each class's Basic count also counts a kill once, even
  with several helpers of that class, and its fraction uses all team kills.
  Class credit, Basic help, and Ultimate help can overlap on one kill.
- Damage/healing done: delivered after modifiers, before health caps, including
  overkill/overhealing. Source/recipient amounts are the sole amount authority;
  source/team/recipient totals derive from them. Regeneration is separate.
- Received healing: delivered Priest healing and actual regeneration, with the
  same combined denominator for component fractions. Excess is allocated among
  simultaneously healing Priests in proportion to delivered healing.
- Controlled recipients: transition-start status, including damage that breaks
  an existing Trap. New same-tick control is not retroactive. Status channels
  overlap and must not be summed as unique controlled time.
- Coordination: mean per-tick Focus Fire Concentration over ticks with at least
  two damaging agents. Single/multiple contributor kill counts and fractions
  count only damage and useful Priest healing on the enemy's death tick. They
  exclude earlier help and appear in both Kill Contributions and Team
  Coordination through shared CSV columns. Focus fire remains about damaging agents.
- Action acceptance: submitted/fully accepted/rejected whole actions. Named
  rejection reasons can overlap and do not add to the number of rejected actions.
- Control/status: application counts belong to the source; active steps belong
  to the affected recipient. Persistent merged effects get no invented caster.
  Time with a slow does not prove the slow reduced movement: Freedom may protect
  the agent. Team time adds each living affected agent's ticks. Two affected
  teammates for one tick count as two agent ticks.
- Trap breaks: damage-broken periods divided by all observed continuous Trap
  periods, including initially active and still-open periods. Two Hunters
  trapping the same enemy on one tick produce two activations but one period.
  A normal later Trap cast damages and breaks the old Trap before applying
  another, or follows natural expiry. It starts a new period. Credit raw-damage
  contributors; measure time left after the timer decreases. Natural expiry
  is not a damage break. Recipient rows describe Traps on that agent, not Traps
  broken by that agent. Team rows describe Traps on enemies of that team.
- Respawn: each team's wave count, mean agents returned per wave, and mean
  observed waiting ticks. Scheduled waves count even when nobody returns, and
  those empty waves stay in the mean agents per wave. Waiting ticks include
  time seen in waits already open at the start or still open at the selected
  tick. Each wait counts equally.
  Dead time remains in Deaths and Time Dead.
- Burst: Mage damage while active, fraction of Mage damage, lethal-tick Burst
  damage, and kill contributions during Burst. Amounts include all damage during
  Burst, not only its extra damage. A source's lethal-tick allocation denominator
  adds its damage to every enemy on the tick that enemy died. Recipient
  contribution uses damage to that one enemy. Shared Deaths counts all abilities.
- Aura coverage: several aura givers can cover the same ally; unique team
  coverage counts that ally once per tick. Coverage fractions use time when
  the giver and ally were alive and had no spawn shield, whether in range or
  not. Allocation compares time covering this ally with all the giver's
  covered ally ticks. Contribution compares that time with all ticks the ally
  had coverage from any teammate of that aura class; these shares can overlap.
  Combined Mage damage gain/Warrior prevention belong to the team, with no
  invented shares for individual aura givers.
- Poison: affected-agent/team **Priest Healing Prevented**; duration reuses status
  steps. Poison must be present at tick start. The value is the reduction in
  delivered Priest healing before the health cap, including healing that would
  have been excess. A heal reduced from 8 to 4 contributes 4 even at full health.
  This value excludes suppressed regeneration. Poison can reduce a Rogue's
  regeneration on its final active tick because the Rogue's combat countdown can
  expire first. Omitting regeneration-prevention and combined-prevention columns
  is a deliberate reporting-scope decision, not an impossible-interaction claim.
- Priest lethal damage rescue: team opportunities, saves and fraction; individual
  rescue participation, including matching Basic and Ultimate shares. Agent
  ability participation uses that Priest's help divided by all unique team saves.
  Team ability fractions count each helped save once and use all unique team saves.
  Capability combines legal available healing under the
  existing masks and Core helpers against actual observed incoming damage;
  individual participation does not imply solo rescue capability.
- Freedom: steps where Freedom reduces the applicable slow restriction, without
  claiming an agent moved or benefited strategically.
- Formation: transition-start living ally distances, with each eligible unordered
  pair counted once. Team mean is weighted by pair observations.

### Historical schema-2 additions

The following 660 measurements were added in schema 2. This table records that
historical addition; the current family table and dictionary include schema 5's
removals, the additions from schemas 6 and 7, schema 10's removal of 46
observed-wait columns, schema 11's recipient columns, schema 12's Basic save
shares, and schema 13's six team ability counts. Schema 10 keeps the two team
means. Schema 4 updated
directed prefixes as described above.

| Addition | Numeric columns | Meaning |
| --- | ---: | --- |
| Effective healing and efficiency | 84 | Source/recipient/team effective amounts and fractions |
| Individual solo kills | 20 | Counts and each agent's solo share of its own kill contributions |
| Observed respawn waits | 48 | Mean dead time, observed periods, unfinished periods and unknown starts |
| Ultimate targeting, output and contributions | 422 | Source/recipient applications, damage/healing, same-tick kill/rescue credit and Burst recipient damage |
| Ultimate healing efficiency | 86 | Effective Ultimate amounts and effective/excess fractions |

Effective Priest healing is delivered healing minus health-cap excess, allocated
proportionally among simultaneous Priests. Source efficiency divides effective or
excess healing by that source's delivered healing; recipient efficiency uses
received Priest healing. Total effective healing received adds actual regeneration.
Effective healing can offset simultaneous damage without increasing health or
ensuring survival. CSV retains real zero amounts for active non-healers, while
the Viewer hides their healing-output rows. Fractions with no delivered healing
are unavailable.

A solo kill has exactly one credited contributor. Useful same-tick Priest support
adds a contributor; healing that is all excess does not. Solo kill fraction divides
the agent's solo kills by its own kill contributions.

Ultimate application fractions divide source/recipient applications by that
source's Ultimate activations. Structurally impossible target relations are
unavailable: Burst applies to its Mage, Priest targets allies, and the other
classes target enemies. Reachable but unused recipients have zero applications;
fractions additionally require a nonzero activation count. Temporary range,
cooldown and control do not alter schema applicability.

Direct Ultimate damage/healing retains delivered amounts; effective Ultimate
healing subtracts its proportional share of excess. Mage Burst deals no immediate
damage; later damage while Burst is active is recorded separately. Ultimate kill
and rescue participation divide the agent's contributions by all team kills or
rescues, respectively. Team counts deduplicate affected enemies or rescued allies.
These values describe observed same-tick contributions, not proof of later causal
effects. Persistent merged status effects have no invented individual caster.

Basic effective and excess healing are accumulated directly over the 50 allied
source–recipient pairs. Subtracting independently rounded total and Ultimate sums
can invent tiny effective healing or erase tiny genuine excess, changing whether
a percentage's denominator is zero. Direct accumulation preserves those meanings
without a tolerance-based zero rule. The existing total and Ultimate statistics
retain their accumulation order.

Every measure has an explicit interpretation in the data dictionary. Directions
refer to comparable conditions and the named effect; they are not universal policy
rankings. For example, lower incoming damage is favorable in isolation, while a
tank may deliberately absorb damage to protect teammates. Target allocation,
formation, activity, opportunity counts and tactical composition remain contextual.
Focus fire uses **Multi-Attacker Ticks**: two damaging agents sharing a target give
100% concentration; attacking different targets gives 50%. It does not measure
attack opportunities. Freedom eligibility requires Freedom, life, no stun and no
spawn shield, including stationary or unslowed ticks. **Ally-Pair Distance
Measurements** counts each living unordered pair: three allies produce three
measurements per executed step.

### Column and presentation audit

Directed families cover Basic/Ultimate ability activations; all/Basic/Ultimate health output;
total/Basic/Ultimate/Burst/solo kill credit; total/Basic/Ultimate rescue credit;
nine named status applications; seven hostile-status damage/healing/kill families;
Trap breaks; and Mage/Warrior aura coverage. Each uses only meaningful pairs.
Controlled families do not add an ability axis. Formation uses 20 unordered ally
pairs. The fixed schema covers all supported rosters, even when an episode leaves
many class-specific columns unavailable.

Application and amount allocation divides a source–recipient quantity by the
source's total of that same quantity. Contribution divides it by the source
team's corresponding quantity for that recipient. Event participation divides
credited contributions by unique qualifying deaths, rescues, or breaks against
that recipient. Pair healing efficiency divides effective or excess healing by
delivered healing on that same pair. Aura coverage rate divides covered by
eligible emitter–beneficiary steps; participation uses unique beneficiary coverage.
All numerators and denominators are exported. The dictionary identifies each.

Unique Basic, Ultimate and class-specific Ultimate event counts are deduplicated
within each transition before accumulation. A kill can involve both a Basic and
an Ultimate contributor, so team Basic events cannot be calculated by subtracting
Ultimate events from all events. Priest support refers to the killed enemy for
kill credit and to the saved ally for rescue credit, separately from the ally
that received the healing application.

Canonical received totals are reused where they already express a team's output
to a recipient. Each canonical measurement has one CSV column. The approved
named status-application columns are an explicit semantic exception: effects
remain directly analyzable even when their counts equal class-filtered ability
uses. Their totals and pairs derive from shared integer application matrices;
there is no second running status-application accumulator. Poison duration reuses
status steps. Both scores/difference and useful count/fraction complements remain.
Full rows copy priority values so each row is independently usable.

The Viewer offers **27 topics**, including the five named Ultimates. Sixteen
topics have separate Totals and By Recipient tables; eleven have one table.
Shared context references the same exported columns once per view. Paired views
share no rows. Totals show teams before acting agents; recipient views show
affected agents before source-to-recipient details. Tooltips identify the exact
CSV name and denominator.
Only structurally applicable rows are displayed, preserving meaningful zero
amounts and undefined ratios. Controls use **Up to Current Tick** and **Entire Episode**.
For example, a Mage cannot heal, so its Healing Done row is hidden. Its CSV
column remains a valid zero, and search still finds it and explains why the
row is hidden. A Priest who has not healed yet keeps its zero Healing Done row.
Measurements about recipients can describe other classes: a Warrior can receive
Priest healing or Burst damage when the relevant team has a capable source.
Ultimate views order uses, relevant output, contributions and persistent effects
within the common totals/detail hierarchy.
One linear prefix index supports seeking without repeated history reductions;
intermediate prefixes never disclose future outcomes or denominators. GUI values
and CSV at the same boundary agree. Inapplicable/undefined cells explain their
missingness. Formation, activity and contextual tactics are **Context dependent**
unless the catalog justifies a stronger interpretation.

## Historical rationale and V1 registry

The sections below preserve design rationale, historical V1 metric IDs and
future-task proposals used by existing artifacts and references. Their older
presentation budgets, standalone metric dispositions and raw-component report
layout do not override the current scalar contract above. V1 artifacts remain
readable; new runs do not retain obsolete cooldown-edge, net-health-change,
clamp-overflow, Trap-interval-end, spawn-shield-expiry, phase-displacement,
combat-countdown-reset, priority-target-share or lifecycle-cause metric groups.
Future KOTH/CTF proposals remain inactive until after manuscript submission.

## Metric constitution

A MARL-BattleGrounds metric must pass all of these tests:

1. A researcher can explain it in one sentence.
2. It answers a real behavioral question rather than merely reporting an
   available statistic.
3. Its denominator represents genuine opportunities.
4. Its direction is clear, or it is explicitly labeled descriptive.
5. It does not claim causality or individual credit that the trajectory does
   not establish.
6. It remains meaningful across algorithms, model architectures, and seeds.
7. Its raw numerator, denominator, counts, sums, durations, and stratification
   keys survive downstream aggregation.
8. It adds material information beyond simpler retained measurements.
9. Its scientific value justifies implementation, compute, storage,
   maintenance, and presentation cost.
10. Its aggregation is statistically valid and its denominator cannot be
    trivially gamed without a visible exposure companion or countermetric.
11. It is replay-verifiable from an authoritative owner without reconstructing
    simulator semantics.

Failure does not always mean deletion. A candidate may move to advanced
description, a controlled scenario, diagnostics, validation-pending research,
or an explicitly rejected ledger.

## Data and interpretation vocabulary

### Health-effect and resolution stages

Every damage, healing, or health-resolution value must use one of these exact
stages:

| Stage | Meaning | Attribution |
| --- | --- | --- |
| `raw_source` | Accepted source output before source and recipient modifiers | Source-aligned |
| `source_modified_gross` | Output after source-side mechanics such as Mage Burst and Mage aura | Source-aligned |
| `recipient_modified_gross` | Effect after recipient-side mitigation or anti-heal, entering simultaneous health resolution | Source-aligned when routing is unique; otherwise recipient total |
| `combat_resolution_health` | Clamped recipient health after simultaneous combat damage/healing and before regeneration | Recipient state, not an effect allocation |
| `realized_net_health_change` | Combat-resolution health minus transition-start health | Recipient only; not uniquely attributable to damage or healing sources |
| `actual_regeneration` | Separately authored post-combat recovery applied after combat resolution | Recipient/lifecycle source |

The word **effective** is not an amount stage. Existing code fields containing
that word are interpreted through the table above.

Milestone 6 CP2 preserves the fixed transition facts losslessly and exposes
sparse events only as a deterministic semantic view. Event absence never
erases false, zero, padded, or continuously active normalized facts. Submitted
and accepted actions remain one normalized authority inside action-acceptance
facts. Evaluation rewards are explicitly `canonical_reward_by_agent` and,
when task-authored, `canonical_reward_by_team`; shaped or auxiliary rewards do
not enter these fields.

The discriminated V1 event union has exactly 24 atomic variants.
`AgentLeftCombatEventV1` records an alive recipient's canonical countdown edge
from one to zero at phase rank 50, after countdown reset and before
regeneration. Newly dead recipient truth belongs to `AgentDiedEventV1`; each
authoritative positive damage source on that lethal transition receives a separate
`LethalDamageContributionEventV1`. Death sorts before its contribution records
at phase rank 90. A contribution is not a killer, last hit, or complete
historical elimination attribution. `TeamDeathmatchScoreChangedEventV1` records
each authoritative positive team-score edge at rank 130, and
`TeamDeathmatchCompletedEventV1` records the authoritative result and completion
basis at rank 140.

Rank-120 ordering uses family-specific coordinates: team waves sort by `(120,
team_index, -1, wave_subtype, neutral_source)` and realized agent respawns by
`(120, configured_team_index, agent_global_slot, respawn_subtype,
neutral_source)`. Each team wave therefore precedes its realized agents, with
teams kept in canonical order.

Recipient-modified gross damage can exceed remaining health. Gross healing can
offset simultaneous damage without producing a positive net-health change.
Consequently, individual realized damage, individual realized healing, and
individual overhealing are undefined without an arbitrary apportionment rule.
MARL-BattleGrounds does not invent one.

### Attribution grades

Every effect-derived metric declares the strongest attribution supported by
the recorded trajectory:

| Grade | Meaning |
| --- | --- |
| `source_exact` | The recorded source-to-recipient route uniquely owns the gross effect or application |
| `recipient_exact` | The recipient outcome is exact, but source allocation is not |
| `unique_emitter_exact` | One eligible emitter uniquely owns a modifier/application in this roster and transition |
| `combined_emitter_set_exact` | The combined team/class effect is exact but division among overlapping emitters is not |
| `attribution_ambiguous` | The requested subject credit is not identifiable and the value is `N/A` |

Canonical non-duplicate rosters can make some aura, anti-heal, Trap, and rescue
credit unique. Duplicate-source rosters fall back to combined team/class
attribution or `N/A`; offline analysis must not invent an apportionment rule.

### Causal-support tiers

| Tier | Meaning | Permitted wording |
| --- | --- | --- |
| `authoritative_outcome` | Direct phase-authored fact or task result | “occurred,” “applied,” “ended because” |
| `deterministic_derived` | Exact transformation of recorded semantic frames/facts | “derived,” “covered,” “was active while” |
| `associational` | Temporal or contextual relation without counterfactual identification | “during,” “followed by,” “associated with” |
| `counterfactual_unsupported` | Requires an unobserved alternate trajectory or arbitrary credit | No official causal claim |

“Within K steps” is an association unless K is a mechanically owned duration
and the claim remains phrased as co-occurrence during that window.

### Subject and symmetry

Every symmetric statistic is computed for Team A and Team B from the same
definition. A difference is exposed only when subtraction is meaningful. Two
teams from one match are paired observations, not independent experimental
units. Agent and class rows are nested descriptive measurements; an absent
class is `N/A`, never zero.

### Disposition axes

Each canonical candidate has four independent labels.

| Axis | Values |
| --- | --- |
| Endpoint role | `primary_confirmatory`, `key_secondary`, `exploratory_descriptive`, `diagnostic_qc`, `rejected` |
| Surface | `primary_team`, `primary_agent_class`, `advanced`, `scenario`, `cross_play_population`, `learning_runtime`, `reward_research`, `none` |
| Readiness | `derivable_now`, `requires_future_task_authority`, `requires_policy_sidecar`, `validation_pending`, `counterfactual_unsupported` |
| Validity | `pass`, `conditional`, `blocked` |

The implementation owner is separately one of: simulator fact, host
evaluation, offline metrics, task, scenario runner, trainer/evaluation harness,
or UI/export. Presentation never owns semantic computation.

### Stable IDs and versioning

IDs use `marlbg.<family>.<metric>.vN`. A change to formula, denominator,
eligibility, subject, direction, amount stage, or attribution semantics creates
a new version. Display-text or formatting-only corrections do not.

One semantic metric may carry long-form dimensions such as team, agent, class,
ability, target class, status channel, task, map, and window. Identical slices
do not become separate metric IDs. Actor-information mode, projection, and
availability remain evidence provenance rather than metric dimensions.

Task mechanics that do not exist yet receive a `candidate.<task>.<name>` key,
not a normative `.v1` metric ID. The owning task replaces that key with a
versioned metric definition only after its authoritative state, events,
eligibility, and edge semantics exist.

### Contract layering

Stable metric semantics must not change merely because a new paper uses a
different opponent pool or confidence interval. The public contract is split
conceptually into four layers:

- **Metric definition:** ID/version, question, scope, data authority, amount
  stage, eligibility, sufficient components, reduction kind, zero-opportunity
  result, attribution, interpretation, gameability, shaping properties, and
  validation state.
- **Evaluation suite:** task, metric selection, endpoint hierarchy, layouts,
  scenarios, rosters, cooperative partners, adversarial opponents, sides,
  frozen joint weights, canonical SharedObs eligibility, completion policy,
  and artifact retention.
- **Experiment manifest:** evaluated algorithms/checkpoints, independent
  training runs, seed schedule, checkpoint selection, comparisons,
  uncertainty method, confidence level, multiplicity family, runtime protocol,
  and train/validation/locked-test partition.
- **Metric result:** metric/suite/manifest identities, complete cell and subject
  coordinates, raw sufficient components, computed value or `null`, result
  status, rollout completion, observer-processing status, per-statistic
  endpoint observation where applicable, and source-schema versions.

Only a semantic-definition change increments a metric version. Population,
weighting, comparison, or inferential changes increment the suite or manifest
version instead. These are documentation contracts in the current milestone,
not a request for a universal production registry.

### Execution-information provenance

[Amendment A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution)
makes execution information invariant provenance for the Paper 1 benchmark,
not a comparison dimension. Every new official metric result is backed by an
episode with mode `shared_obs`, actor projection
`base-observation-plus-authorized-sensor-source-bank@1`, and the exact
configured-active, same-team, off-diagonal availability matrix on every replay
frame. This provenance is retained so consumers can prove eligibility; it does
not create separate metric IDs, suite strata, cell weights, leaderboard rows,
or matchup directions.

The expected matrix comes from the frozen configured roster, not living state,
health, visibility, policy identity, or frame index. Configured-but-dead sources
remain authorized while their ordinary sensor material remains
lifecycle-zeroed. Exact all-frame equality rejects an incomplete teammate
topology and mid-episode drift; an all-false matrix is canonical only when the
roster contains no same-team off-diagonal pair.

`EvaluationEpisodeContextV1` retains one episode-global
`execution_information_mode` and one episode-global `actor_projection`.
Generic/custom V1 evaluation and replay validation continue to support both
homogeneous SharedObs and homogeneous NoSharedObs, and V1 still cannot encode a
mixed per-slot assignment by overloading its global mode, policy ID, or false
availability rows. NoSharedObs evidence remains valid for diagnostics, custom
research, and historical replay compatibility, but it is ineligible as a new
official metric result. The current roadmap contains no mixed-regime V2 work.

## Presentation budgets

### Primary team card

Each task receives at most four endpoint blocks under the canonical SharedObs
eligibility contract; this is a ceiling, not a quota:

1. win/draw/loss as one outcome distribution;
2. terminal canonical score differential;
3. one non-redundant task-native signature; and
4. at most one validated coordination descriptor, displayed with its exposure.

The footer always states independent training seeds, evaluation cells,
episodes, failures/truncations, cell weighting, and uncertainty method.

### Primary agent/class card

The compact card contains four universal descriptive blocks and at most one
class signature:

- recipient-modified gross damage output and team share;
- recipient-modified gross healing output and team share, otherwise `N/A`;
- recipient-modified gross damage received and team share;
- deaths and team death share; and
- one validated class signature.

These columns describe role usage. They do not create a universal ranking of
heterogeneous classes. Lethal-transition contribution remains available in the
advanced export, but it is too outcome-local and gameable to be a universal
primary role fact.

### Advanced, scenario, and diagnostic surfaces

The advanced export preserves tidy long-form observations and raw sufficient
statistics. It favors distributions, medians, and interquartile ranges over
default min/max columns. Scenario cards contain one primary quantitative
matched full-method-versus-ablation contrast, at most two supporting secondary
margins, explicit violations, and replay. Scenario cards describe controlled
behavior under their frozen conditions; they are not general-strength scores
and never contribute to Elo. Diagnostics remain outside tactical leaderboards.

Every displayed metric exposes its ID/version, units or health-effect stage,
subject, direction or descriptive label, opportunity and `N/A` behavior,
aggregation protocol, defined/undefined counts, and concise allowed
interpretation. Tooltips are reachable by hover, keyboard focus, and click.
Sortable columns use one consistent interaction and preserve team/opponent
labels plus canonical mode/projection provenance. CSV and JSON export tidy raw
sufficient components, not only rounded dashboard values. UI/export code
references this contract and does not own a second formula.

## Raw sufficient-component defaults

This document owns what each semantic metric must preserve; the
[evaluation protocol](protocol.md) exclusively owns reduction order, cell
weighting, inferential units, and uncertainty. Unless a metric row states
otherwise, preserve:

- a total's sum and eligible episode count;
- a rate's numerator, genuine-opportunity denominator, and zero-opportunity
  count;
- a share's subject component and corresponding team total;
- a duration's qualifying and eligible agent-steps; and
- a distribution's event- or episode-level observations at its declared unit.

A zero opportunity produces `N/A`, not zero. Partial prefixes are excluded
from every official endpoint estimator unless the metric explicitly declares
prefix validity. Scientific censoring is a separate per-statistic endpoint
observation, not an episode completion state. Prefix-valid diagnostic or
descriptive components may be exported only with rollout and processing status
and remain outside the official estimator.

Result status follows this precedence: `invalid_artifact`,
`structurally_inapplicable`, `ambiguous_attribution`, `insufficient_data`,
`zero_opportunity`, then `defined`. A currently zero denominator on an
ineligible prefix is insufficient data rather than zero opportunity. A
right-censored result is defined only when that versioned metric declares a
censoring estimand and preserves the required censoring component.

Milestone 6 CP3 supplies strict generic count, sum, ratio-component,
duration-component, opportunity, and distribution-observation records. It does
not implement the metric formulas below. Historical records preserve raw
components and complete semantic keys; ratios, means, ratings, uncertainty,
and presentation values remain downstream derivations. Agent and policy
subjects must join configured-active context rows. An absent class may appear
as `structurally_inapplicable`, never as a fabricated zero, and padded actors
never enter opportunity denominators merely because their canonical no-op mask
has a valid category.

A historical CP3 draft is episode-local. Its `eligible_episode_count` is
therefore exactly `0` or `1`, and a ratio's `zero_opportunity_occurrence` records
the final episode-level `0` or `1` incidence. Cross-episode interpretation must
preserve these finalized raw rows and their zero-opportunity episodes.
At final materialization, an incomplete or otherwise ineligible row carries
`eligible_episode_count = 0`; `defined` and eligible `zero_opportunity` rows
carry `1`. A ratio with a recorded zero-opportunity occurrence finalizes as
`zero_opportunity`, never `defined`. Other zero-valued component families do
not by themselves reveal metric-specific opportunity semantics. A separately
recorded observer-processing failure does not erase a fully consumed prefix:
complete-only eligibility requires rollout completion and equal validated and
processed transition counts, not a successful-status label by itself.

For standard replay, `EvaluationMetricReportV1` is wrapped in its own
content-addressed artifact and joined to the replay's pre-link trajectory
content. The replay stores only the report artifact's path-free identity,
schema, digest, and canonical byte length. Missing sidecar bytes do not make
the semantic trajectory unrenderable, but they do make metric-bundle evidence
incomplete and must be reported as such. See the
[standard replay format](replay_format.md).

## Canonical task-independent metrics

### Outcome and completion templates

| ID | Question and sufficient statistic | Role / surface | Direction | Readiness / owner |
| --- | --- | --- | --- | --- |
| `marlbg.task.outcome_distribution.v1` | How often did the team win, draw, or lose? Preserve the three mutually exclusive counts and total eligible games. | `primary_confirmatory` / `primary_team` | win higher, loss lower; one multinomial endpoint | `derivable_now` / host evaluation |
| `marlbg.task.terminal_score_differential.v1` | By how much did the team lead at the terminal frame? Preserve `team_score - opponent_score` per complete episode. | `primary_confirmatory` / `primary_team` | higher | `derivable_now` / host evaluation |
| `marlbg.task.evaluation_return.v1` | What canonical evaluation reward did the team/agent receive? Preserve the unshaped episode return and reward-mode identity. | `key_secondary` / `advanced` | task-defined | `derivable_now` / host evaluation |
| `marlbg.task.episode_length.v1` | How many valid transitions occurred before completion? Preserve transition count and end reason. | `exploratory_descriptive` / `advanced` | descriptive | `derivable_now` / host evaluation |
| `marlbg.artifact.completion.v1` | Was the rollout complete, partial, interrupted, or failed, and did host processing succeed? Preserve completion basis, validated/processed prefix lengths, processing failure, and end/failure reason. | `diagnostic_qc` / `none` | descriptive | `derivable_now` / host evaluation |

Outcome rows are complete-only. A missing outcome, failed run, or
right-censored endpoint never silently becomes a draw, loss, zero, or excluded
row. Truncation alone does not imply a result; Team Deathmatch is the explicit
exception where an authoritative horizon completion event supplies a draw.

### Combat output and exposure

| ID | Exact definition and raw components | Role / surface | Direction | Validity and caveat |
| --- | --- | --- | --- | --- |
| `marlbg.combat.recipient_modified_gross_damage_output.v1` | Sum, by source/team, of source-modified gross damage times the recipient modifier for accepted routed damage. Store HP and Basic/Ultimate dimensions. | `key_secondary` / primary cards + advanced | descriptive | `pass`; not realized HP loss |
| `marlbg.combat.recipient_modified_gross_damage_received.v1` | Sum authoritative recipient-modified gross damage by recipient/team. | `key_secondary` / primary cards + advanced | descriptive | `pass`; exposure is not automatically poor positioning |
| `marlbg.support.recipient_modified_gross_healing_output.v1` | Sum, by source/team, of source-modified gross healing times the recipient modifier for accepted routed healing. | `key_secondary` / primary cards + advanced | descriptive | `pass`; do not label “effective healing” or realized restoration |
| `marlbg.combat.realized_net_health_change.v1` | For each recipient-transition, preserve post-combat health minus transition-start health, plus gross damage and healing. | `exploratory_descriptive` / advanced | descriptive | `pass`; recipient-only net result |
| `marlbg.combat.upper_health_clamp_overflow.v1` | Recipient-level positive health amount discarded only by the upper health clamp after simultaneous netting. Preserve overflow and gross-healing exposure. | `exploratory_descriptive` / advanced | lower may indicate less saturation, but conditional | `conditional`; never attribute to one healer under overlap |
| `marlbg.combat.death_count.v1` | Count authoritative `AgentDiedEventV1` newly dead recipients for each team and class. | `key_secondary` / primary cards + advanced | lower for own team, higher for opponent, task-context dependent | `pass` |

Shares use the pooled agent/class component divided by its pooled team total
within the same cell. A team total of zero yields `N/A`. Damage/healing totals
are always accompanied by episode count and exposure opportunities.

### Lethal-transition damage contribution and coordinated offense

| ID | Exact definition and raw components | Role / surface | Direction | Validity and caveat |
| --- | --- | --- | --- | --- |
| `marlbg.combat.lethal_transition_damage_contribution.v1` | Count atomic `LethalDamageContributionEventV1` records where the source dealt authoritative positive recipient-modified gross damage on an enemy's lethal transition. Preserve source, recipient, class, team, and team enemy-death count. | `exploratory_descriptive` / advanced | descriptive | `pass`; not a kill, last hit, or complete historical elimination contribution |
| `marlbg.combat.lethal_transition_contribution_rate.v1` | Lethal-transition damage contributions divided by enemy deaths caused by the source's team, using the protocol's raw-component reduction. | `exploratory_descriptive` / advanced | descriptive | `conditional`; `N/A` when the team caused no deaths and susceptible to last-transition crowding |
| `marlbg.coordination.single_contributor_lethal_transition_count.v1` | Count enemy deaths having exactly one authoritative positive-damage source on the lethal transition. | `exploratory_descriptive` / advanced | descriptive | `pass`; earlier damage may have occurred, so this is not a “solo kill” |
| `marlbg.coordination.multi_contributor_lethal_transition_rate.v1` | Enemy deaths with at least two allied positive-damage sources on the lethal transition divided by all team-caused enemy deaths; always show the denominator. | `key_secondary` / primary-team candidate + advanced | descriptive | `conditional`; outcome-conditioned and gameable without death exposure |
| `marlbg.coordination.focus_fire_concentration.v1` | On a transition with at least two allied positive-damage sources, let `n_r` be sources damaging enemy recipient `r`; retain `max_r(n_r) / sum_r(n_r)` and one opportunity. Pool sum and opportunity count. | `key_secondary` / primary-team candidate + advanced | descriptive | `conditional` until construct validation; no eligible transition gives `N/A` |

Focus-fire concentration measures same-transition target concentration, not
whether the chosen target or degree of concentration was strategically
correct. It may enter a primary card only after blinded replay validation and
counterexample review.

### Abilities, cooldowns, and movement

| ID | Exact definition and raw components | Role / surface | Direction | Readiness / caveat |
| --- | --- | --- | --- | --- |
| `marlbg.ability.activation_count.v1` | Count accepted Basic and Ultimate activations by source/class/ability; rejected submissions are separate. | `exploratory_descriptive` / advanced | descriptive | `derivable_now` |
| `marlbg.ability.ultimate_cooldown_start_count.v1` | Count accepted actions with Ultimate enabled. | `exploratory_descriptive` / advanced | descriptive | `derivable_now`; no cooldown fact leaf |
| `marlbg.ability.ultimate_ready_transition_count.v1` | Count adjacent-frame positive-to-zero Ultimate cooldown edges. | `exploratory_descriptive` / advanced | descriptive | `derivable_now`; no cooldown fact leaf |
| `marlbg.movement.phase_displacement.v1` | Preserve exact vector and distance distributions separately for Charge-phase and ordinary-movement-phase realized displacement. | `exploratory_descriptive` / advanced | descriptive | `derivable_now` after M6 CP1; both vectors are phase-authored without a second geometry pass and are not positioning-quality scores |

Cooldown use rates require a task/experiment-owned opportunity definition.
Readiness alone is not an instruction to activate, so “ultimate efficiency” is
not a universal quality metric.

### Status and crowd-control lifecycle

| ID | Exact definition and raw components | Role / surface | Direction | Validity and caveat |
| --- | --- | --- | --- | --- |
| `marlbg.status.application_count.v1` | Count authoritative applications by source, recipient, and stable status channel. | `exploratory_descriptive` / advanced | descriptive | `pass` |
| `marlbg.status.active_recipient_steps.v1` | Count frame-level active status recipient-steps; preserve eligible active/alive recipient-steps and channel. | `exploratory_descriptive` / advanced | descriptive | `pass`; “uptime” is the derived ratio |
| `marlbg.status.lifecycle_cause_count.v1` | Count age-to-zero, refresh/extension, damage-break, and new-death-clear causes independently by recipient/channel. | `exploratory_descriptive` / advanced | descriptive | `pass`; causes may coexist |
| `marlbg.control.damage_to_controlled_recipient.v1` | Recipient-modified gross damage routed while the recipient had a named transition-start control status. | `exploratory_descriptive` / advanced | descriptive | `pass`; context association, not proof of follow-up quality |
| `marlbg.control.enemy_death_while_controlled.v1` | Enemy deaths occurring while a named transition-start control status was active. | `exploratory_descriptive` / advanced | descriptive | `pass`; associational |

Do not sum heterogeneous slow, stun, anti-heal, Burst, and Freedom durations
into one “CC score.” Channel-level durations and exact lifecycle causes remain
available in the long-form export.

### Class-specific retained metrics

| ID | Exact definition and raw components | Role / surface | Direction | Attribution and readiness |
| --- | --- | --- | --- | --- |
| `marlbg.mage.burst_window_damage.v1` | Recipient-modified gross damage from the Mage while its Burst status is active; preserve activations, active Mage-steps, and damage exposure. | `key_secondary` / primary class candidate + advanced | descriptive | `deterministic_derived`; damage during Burst, not necessarily caused by activation |
| `marlbg.mage.burst_enemy_death_association.v1` | Enemy deaths where the Mage dealt positive damage on the lethal transition while Burst was active. Preserve Burst activations and team enemy deaths. | `exploratory_descriptive` / advanced | descriptive | associational; not “kills caused by Burst” or complete historical contribution |
| `marlbg.mage.aura_coverage.v1` | For each Mage emitter, covered eligible emitter-beneficiary steps divided by all same-team steps where both emitter and beneficiary are active, alive, and unshielded. Preserve emitter/beneficiary counts. | `key_secondary` / primary class candidate + advanced | descriptive | exact per-emitter coverage after M6 CP1 |
| `marlbg.mage.combined_aura_amplification.v1` | Combined recipient-modified gross damage increment attributable to the recorded Mage-aura multiplier, aggregated by team/class. | `exploratory_descriptive` / advanced | descriptive | exact combined value; per-emitter value blocked when emitters overlap |
| `marlbg.warrior.aura_coverage.v1` | For each Warrior emitter, covered eligible emitter-beneficiary steps divided by all same-team steps where both emitter and beneficiary are active, alive, and unshielded. | `key_secondary` / primary class candidate + advanced | descriptive | exact per-emitter coverage after M6 CP1 |
| `marlbg.warrior.combined_aura_mitigation.v1` | Sum pre-recipient damage minus recipient-modified gross damage where the recorded Warrior aura modifier applies. | `key_secondary` / primary class candidate + advanced | descriptive | exact combined team/class value; per-emitter value blocked under overlap |
| `marlbg.hunter.trap_active_steps.v1` | Active Hunter Trap stun recipient-steps with application and eligible-recipient exposure. | `key_secondary` / primary class candidate + advanced | descriptive | exact; replaces placed-trap uptime terminology |
| `marlbg.hunter.trap_status_episode_end.v1` | Segment Trap status episodes from authoritative lifecycle causes in simulator phase order; record whether each completed episode ended by age, damage break, or death clear. An ordinary refresh without an end remains in the same episode. An end followed by same-transition reapplication closes the old episode and starts a new one even though no zero-duration frame exists. | `exploratory_descriptive` / advanced | descriptive | exact after M6 CP1; multiple episode edges may occur in one transition and artifact-end censoring is explicit |
| `marlbg.hunter.trap_damage_break_rate.v1` | Completed Trap status episodes ending by damage divided by all completed Trap status episodes; report censored active episodes separately. | `key_secondary` / primary class candidate + advanced | descriptive | conditional; never use casts as a silent denominator |
| `marlbg.rogue.combined_anti_heal_reduction.v1` | Sum source-modified healing minus recipient-modified gross healing while anti-heal applies. Preserve healing exposure and active steps. | `key_secondary` / primary class candidate + advanced | descriptive | exact combined reduction; per-Rogue credit blocked under overlap |
| `marlbg.rogue.priority_target_damage_share.v1` | Recipient-modified gross Rogue damage to a task-declared priority-target class/state divided by all Rogue gross damage. | `exploratory_descriptive` / advanced | descriptive | target class available now; flag-carrier state requires CTF authority |
| `marlbg.priest.same_transition_lethal_damage_rescue.v1` | Count living recipients where gross damage alone reached/exceeded start health, gross healing was positive, and post-combat health remained positive. Preserve recipient opportunities and unique/multiple healer count. | `key_secondary` / primary class candidate + advanced | higher with exposure companion | recipient/team causal outcome; source attribution only when exactly one healer contributed |
| `marlbg.priest.freedom_binding_coverage.v1` | Among active, alive, unshielded, unstunned recipient-frames with Freedom active, the fraction satisfying `max(canonical_float32_slow_product, global_slow_floor) < freedom_floor`. The product uses the catalog's stable slow-channel order. Preserve binding and eligible frames separately; return `N/A` at zero eligibility. Frame `t` governs movement in `t -> t+1`; stay actions remain eligible. | `exploratory_descriptive` / advanced | descriptive | exact frame/catalog derivation; permits only “Freedom was mechanically binding,” not distance recovered, movement caused, or tactical value |

No class signature is guaranteed a primary slot merely because it is
derivable. Each must pass construct validation on realistic 5v5 replays and
must remain useful when duplicate-class ablations are enabled.

For Trap episode reconstruction, lifecycle causes are consumed in their
authoritative order. Age or damage break closes the currently active episode;
application can then start a new episode on the same transition; new-death
clearing can close that newly applied episode immediately. Thus
break–reapplication–death-clear may yield two completed episodes in one
transition. One episode end carries the set of coexisting authoritative causes
rather than a forced precedence; it contributes once to the completed-episode
denominator and to the damage-break numerator when that set contains
damage-break. An episode still active at artifact end is right-censored and is
reported separately from the completed-episode denominator.

### Lifecycle and diagnostic metrics

| ID | Exact definition and raw components | Role / surface | Direction | Readiness / caveat |
| --- | --- | --- | --- | --- |
| `marlbg.recovery.realized_regeneration.v1` | Sum actual post-clamp out-of-combat regeneration separately from combat healing. | `exploratory_descriptive` / advanced | descriptive | `derivable_now` |
| `marlbg.recovery.combat_countdown_reset_count.v1` | Count authoritative combat-countdown resets by agent and preserve current/next countdown context. | `diagnostic_qc` / advanced | descriptive | lifecycle diagnostic; not an engagement boundary |
| `marlbg.lifecycle.respawn_wave.v1` | Count authoritative team waves and realized agent respawns, preserving team, agent, and countdown context. | `diagnostic_qc` / advanced | descriptive | lifecycle diagnostic, not policy quality by itself |
| `marlbg.lifecycle.dead_agent_steps.v1` | Count configured active frame-agent rows that are dead; preserve new-death, respawn, and episode-end boundaries for life/dead-duration distributions. | `exploratory_descriptive` / advanced | descriptive | an unrespawned terminal life/dead interval is censored for duration analysis |
| `marlbg.lifecycle.spawn_shield_expiry.v1` | Count ordinary shield expiries and preserve active-at-start exposure. | `diagnostic_qc` / advanced | descriptive | lifecycle diagnostic |
| `marlbg.diagnostic.action_acceptance_rate.v1` | Configured active actor-transitions with no tuple-domain, movement, or combat-pair rejection divided by all configured active actor-transitions; preserve per-component accepted/rejected counts. | `diagnostic_qc` / none | higher usually indicates healthier policy plumbing | dead/stunned canonical choices remain real masked decisions; padded slots are excluded |
| `marlbg.diagnostic.action_rejection.v1` | Count submitted tuple-domain, movement-mask, and combat-pair rejection facts with submitted/accepted actions and opportunity count. | `diagnostic_qc` / none | lower usually indicates healthier policy plumbing | no invented LOS/range/cooldown reason |
| `marlbg.formation.ally_distance_distribution.v1` | Distribution of pairwise distances between eligible active/alive allies from semantic frame positions. | `exploratory_descriptive` / advanced | descriptive | not a cohesion-quality score; no new core distance fact |

Time dead, death-to-respawn duration, life duration, wave size, countdown
history, and recovery timing remain available as lifecycle diagnostics when a
named analysis needs them. They do not enter the primary tactical scorecard.

## Future and pending task-owned metrics

These dispositions are stable, but each row remains inactive until its named
activation dependency exists. Team Deathmatch score/outcome authority and its
episode reducers are implemented; the candidate below remains inactive.

### Team Deathmatch

| ID | Required definition | Role / surface | Activation dependency |
| --- | --- | --- | --- |
| `candidate.tdm.team_wipe_count` | Task-defined transitions or intervals where every eligible opposing agent is dead; preserve respawn-wave context. | `exploratory_descriptive` / advanced | separate research activation with exact interval semantics |

TDM does not create individual killer ownership, agent K/D, or generic teamfight
victories. A36 separately permits a descriptive pooled team K/D ladder column:
sum authoritative team score increments and opposing score increments across
matches, accounting for nonzero initial scores, and divide only when total deaths
are positive. This creates no individual kill attribution or rating input.
Elimination differential is omitted because official TDM scoring makes it
mathematically identical to terminal score differential. Team-wipe count stays
inactive until a separate research need justifies exact interval semantics.

TDM is threshold-victory: reaching the configured threshold is the only route
to a win. If neither team reaches it by the horizon, the authoritative outcome
is a draw regardless of terminal score differential. If both teams cross on
one simultaneous transition, the complete successor scores decide the result;
an equal score draws. Terminal score differential remains descriptive evidence,
not an alternative horizon winner rule.

### Three-hill King of the Hill

| ID | Required definition | Role / surface | Activation dependency |
| --- | --- | --- | --- |
| `candidate.koth.eligible_hill_control_share` | Team-controlled eligible hill-timesteps divided by all eligible team-control hill-timesteps; preserve neutral and contested states separately. | `primary_confirmatory` / primary team | authoritative per-hill control state |
| `candidate.koth.contest_share` | Contested eligible hill-timesteps divided by all eligible hill-timesteps. | `exploratory_descriptive` / advanced | authoritative contest state |
| `candidate.koth.control_transition_outcome` | Versioned counts of neutral/enemy/ally control transitions, including defense retention and capture/steal outcomes defined by the task. | `key_secondary` / advanced | task-authored transition events |
| `candidate.koth.hill_occupancy` | Per-agent/class eligible occupancy steps by ally/neutral/enemy hill and control state. | `exploratory_descriptive` / advanced | authoritative membership state |
| `candidate.koth.allocation_profile` | Long-form distribution of eligible agents across the three stable hill identities; no universal higher-is-better entropy score. | `exploratory_descriptive` / advanced | task and layout identities |

Control, contest, and occupancy use hill-timesteps, not bare episode horizon.
There is no arbitrary per-agent score contribution and no deaths/score
“objective fight cost” ratio.

### Capture the Flag

| ID | Required definition | Role / surface | Activation dependency |
| --- | --- | --- | --- |
| `candidate.ctf.capture_count` | Authoritative captures by team. | `primary_confirmatory` / primary team | flag/capture events and score |
| `candidate.ctf.capture_conversion` | Captures divided by eligible enemy-flag pickups, always displayed with that pickup exposure. | `primary_confirmatory` / primary team | pickup/capture identity |
| `candidate.ctf.forced_drop_interception` | Task-defined enemy-carrier forced drops/interceptions divided by eligible opposing carry episodes; preserve raw episodes and causes. | `key_secondary` / primary candidate + advanced | carrier/drop cause semantics |
| `candidate.ctf.friendly_return` | Task-defined friendly returns divided by eligible friendly dropped-flag episodes; report auto-return separately. | `key_secondary` / advanced | return causes |
| `candidate.ctf.flag_possession` | Carrier agent-steps by team/class and share of eligible possession time. | `exploratory_descriptive` / advanced | carrier state |
| `candidate.ctf.escort_coverage` | Carrier steps with at least one eligible allied non-carrier in a declared escort relation divided by carrier steps. | `exploratory_descriptive` / advanced | versioned relation and carrier state; not escort quality |

Carry time by class is descriptive. “Preferred carrier class share,” arbitrary
agent score credit, and universal escort/interception quality are rejected;
decision quality belongs in controlled scenarios.

On the compact team card, capture count, pickup-to-capture conversion, and the
pickup exposure supporting that conversion form one compound CTF task-native
block. They do not consume multiple task-signature slots or become independent
confirmatory families merely because all raw components are visible.

## Scenario-owned behavior matrix

Scenario metrics are designed primarily for matched behavioral ablations. The
evaluation definition binds one full method and one declared ablation to the
same content-addressed scenario, deterministic pressure controller, canonical
SharedObs contract, seeds, side assignments, training budget,
checkpoint-selection rule, and endpoint. One predeclared primary behavioral
contrast answers the claim; no more than two secondary margins may support it.
Differences outside the declared ablation invalidate causal interpretation.

The pressure binding may use the general Reactive TDM policy or a specialist
such as Scenario 3's body-aware Rogue. Its exact behavior identity remains an
evaluation condition, not a metric or baseline-strength claim. DevClient
availability and physical regression results alone do not qualify a scenario,
define its endpoint, or establish its horizon. Scenario-derived rules remain
within the protected evaluation content closure; their originating results
cannot be represented as uncontaminated evaluation of those engineered rules.

| Behavior | Why no episode-wide quality scalar | Required scenario evidence |
| --- | --- | --- |
| Peeling / backline protection | Value depends on threat, protected ally, and resulting trade | protected-ally survival or health margin; threat displacement/control; violations |
| Kiting / disengagement / re-engagement | Low damage taken can also mean non-participation | survival or damage-trade endpoint under a fixed pursuer; distance/time margin; participation constraint |
| Flanking / backline access | Geometry and timing make angle alone ambiguous | priority-target access/effect endpoint; time or health margin; route/visibility constraints |
| Body blocking / escape denial | Contact alone may be accidental or harmful | protected route or interception outcome; progress margin; collision/position replay |
| Healing triage | Correct recipient depends on synchronized threats | weighted survival/health endpoint under fixed simultaneous pressure; response margin; invalid-target violations |
| Regrouping | Fast cohesion can be strategically wrong | survival/objective readiness after a fixed respawn split; time margin; premature-engagement violations |
| Rotations / multi-objective allocation | Entropy and rotation count have no universal direction | objective conversion under fixed multi-hill pressure; score/time margin; overcommit violations |
| Escort / interception | Proximity does not establish useful protection | capture/stop endpoint under fixed carrier route; completion/censored time; role violations |
| Trap discipline | Aggregate uptime cannot identify correct tactical timing | fixed threat-specific denial/peel endpoint; control/follow-up margin; misuse violations |
| Burst synchronization | Damage during Burst does not prove good timing | fixed coordinated damage/objective endpoint; activation timing margin; survival/position constraint |
| Freedom-assisted movement | Exact counterfactual movement is not in the trajectory | fixed slowed traversal/rescue endpoint; completion/progress margin; status evidence |

Every scenario must follow the scenario protocol. Replay is mandatory evidence
but never substitutes for the quantitative endpoint. Multiple independently
trained, deliberately paired treatment/control runs support inference; the
training run—not an agent, tick, episode, or team—is the replication unit.
Scenario results provide bounded evidence under frozen conditions. They neither
prove general competence nor enter ratings, checkpoint selection, reward
shaping, curricula, population weights, or any other adaptive process.

## Population, learning, and runtime metrics

| ID | Definition | Role / surface | Owner |
| --- | --- | --- | --- |
| `marlbg.population.matched_partner_performance.v1` | Task performance with the declared training-related cooperative partner under a frozen adversarial-opponent distribution. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.held_out_partner_performance.v1` | The same task endpoint with a disjoint cooperative-partner pool while holding the adversarial-opponent distribution fixed. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.held_out_opponent_performance.v1` | The same task endpoint against a disjoint adversarial-opponent pool while holding the cooperative-partner distribution fixed. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.partner_generalization_gap.v1` | Matched-partner minus held-out-partner performance under the same opponent panel and frozen joint cell weights; always report both absolute components. | `exploratory_descriptive` / cross-play population | evaluation harness |
| `marlbg.population.lower_tail_performance.v1` | Predeclared lower quantile over one explicitly named partner or opponent population dimension, never an unstable unqualified minimum. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.rating.v1` | Secondary rating with pool, protocol, side assignments, rating-system version, and a complete authoritative raw outcome matrix. | `exploratory_descriptive` / cross-play population | evaluation harness |
| `marlbg.learning.fixed_budget_performance.v1` | Official evaluation endpoint at a predeclared budget; report both environment transitions and active-agent decision transitions. | `primary_confirmatory` / learning runtime | trainer/evaluation harness |
| `marlbg.learning.curve_auc.v1` | Area under a predeclared held-out evaluation curve with fixed x-axis, horizon, checkpoint schedule, and interpolation rule. | `key_secondary` / learning runtime | trainer/evaluation harness |
| `marlbg.learning.time_to_threshold.v1` | First predeclared evaluation checkpoint reaching a frozen threshold; non-reaching runs remain censored. | `exploratory_descriptive` / learning runtime | trainer/evaluation harness |
| `marlbg.runtime.environment_throughput.v1` | Environment transitions per second under a versioned hardware/batch/JIT protocol. | `diagnostic_qc` / learning runtime | trainer/runtime harness |
| `marlbg.runtime.policy_inference.v1` | Policy inference latency/throughput under the same declared measurement protocol. | `diagnostic_qc` / learning runtime | trainer/runtime harness |

Cross-play summaries never discard the underlying focal-by-partner-by-opponent
tensor. Matched, held-out, cooperative-partner, and adversarial-opponent
experiments are distinct populations and cannot share an unlabeled
“robustness” number.

For the planned Big 12 instantiation, exactly twelve method entrants contribute
one validation-selected fixed system and one Elo value each. The resulting 66
unordered pairings contain 100 episodes apiece—five maps by ten evaluation
coordinates by two side assignments—for 6,600 episodes total. The complete
win/draw/loss matrix is authoritative. The qualified compact estimator is one
jointly fitted, draw-aware Bradley–Terry–Davidson model centred at 1200, and is
secondary to that matrix. The evaluation protocol freezes its
[parameterization, uncertainty, convergence, failure rules and presentation](protocol.md#frozen-rating-and-uncertainty-contract).
The implementation is qualified on bounded synthetic tournaments; manuscript
policy training and the full Paper 1 tournament remain future work.

Rows 1–11 each retain three independent training runs. A rule frozen before
training first selects an eligible checkpoint within each run from validation
information alone, then selects the validation-highest of those three as the
method's sole tournament system. Locked scenarios and tournament outcomes may
not participate. Qwen-Five remains tentative until it passes a measured cost,
throughput, reproducibility, and compatibility gate. Weekly Big 12 ratings are
identified by immutable roster snapshot and are not directly comparable across
changing pools without a separately specified longitudinal model.

## Reward-shaping classification

Metric definitions do not automatically become rewards. A single label would
conflate availability, information privilege, credit, and objective impact, so
future M10 tooling records four independent axes:

| Axis | Values |
| --- | --- |
| Availability | `transition_local`, `wrapper_memory`, `episode_terminal`, `offline_only` |
| Information | `actor_observable`, `shared_obs_observable`, `centralized_training_privileged`, `external_annotation` |
| Credit scope | `team`, `agent`, `pair`, `population` |
| Objective effect | `potential_based_candidate`, `objective_changing`, `unknown`, `unsuitable` |

Each proposed component additionally records whether it is available on the
JAX hot path and its known reward-hacking risks.

The following classification covers every current retained family. It states
what an experiment could access, not what MARL-BattleGrounds recommends as a
default reward. In particular, `shared_obs_observable` remains a semantic
information-availability classification; A25 does not rename or erase it.

| Metric IDs | Availability | Information | Credit | Objective effect and principal risk |
| --- | --- | --- | --- | --- |
| `marlbg.task.outcome_distribution.v1`, `marlbg.task.terminal_score_differential.v1`, `marlbg.task.evaluation_return.v1` | `episode_terminal` | `centralized_training_privileged` | team | `objective_changing`; can duplicate or replace canonical task intent |
| `marlbg.task.episode_length.v1`, `marlbg.artifact.completion.v1` | `episode_terminal` | `external_annotation` | team | `unsuitable`; optimizing duration/completion can reward premature endings or infrastructure artifacts |
| gross damage/healing, realized health, clamp overflow, and death-count IDs | `transition_local` | `centralized_training_privileged` | agent or team | `objective_changing`; throughput farming and suicidal trades require outcome companions |
| lethal-transition contribution and single-/multi-contributor IDs | `transition_local` | `centralized_training_privileged` | agent or team | `objective_changing`; outcome-conditioned credit can encourage last-transition crowding |
| `marlbg.coordination.focus_fire_concentration.v1` | `transition_local` | `centralized_training_privileged` | team | `unknown`; concentration is gameable and target quality is absent |
| activation and cooldown IDs | `transition_local` | `actor_observable` for own values | agent | `objective_changing`; casting or holding is not inherently good |
| `marlbg.movement.phase_displacement.v1` | `transition_local` | `centralized_training_privileged` | agent | `unknown`; distance can reward purposeless motion |
| status application/lifecycle and controlled-recipient IDs | `transition_local`; `wrapper_memory` only for longer windows | `centralized_training_privileged` | agent, pair, or team | `objective_changing`; raw uptime/application can reward low-value control |
| Mage, Warrior, Hunter, Rogue, and Priest class IDs | `transition_local`; `wrapper_memory` for status episodes | `centralized_training_privileged` | agent or team | `objective_changing` or `unknown`; each requires its exposure and task outcome companions |
| regeneration, respawn-wave, spawn-shield, action-rejection, and ally-distance IDs | `transition_local` | privileged facts/frames or `external_annotation` for QC | agent or team | `unsuitable` for canonical shaping; easily rewards stalling, dying, rejection suppression, or arbitrary spacing |
| future TDM/KoTH/CTF candidate keys | task-owned after implementation | task-dependent | team, agent, or pair | `unknown` until task reward and opportunity semantics are authoritative |
| scenario-owned behavior | `offline_only` | `external_annotation` plus replay | scenario subject | `unsuitable`; official scenario material and results are prohibited from reward shaping, curriculum design, or any other adaptive use |
| population, learning, and runtime IDs | `offline_only` | `external_annotation` | population | `unsuitable`; not transition credit signals |

“Actor observable” means the actor legitimately receives the required
same-epoch or adjacent own values; it does not authorize privileged team facts
as actor input. Any implementation promotes an ID to a shaping component only
through its own versioned configuration and reward-hacking review.

The canonical task reward and each named shaping component remain separately
logged. Host Pydantic models, sparse events, metric formulas, and replay files
never feed the training hot path.

## Consolidated disposition index

This index consolidates aliases from the design PDF and metric brainstorm. The
private M6 source ledger preserves the line-level source trace.

| Candidate family or alias | Final disposition | Canonical replacement or reason |
| --- | --- | --- |
| Win/loss/draw, win rate | Retain, primary | One multinomial outcome distribution |
| Score, score differential, final margin | Retain one dominance endpoint | Terminal score differential; omit redundant fields per task |
| Return | Retain, secondary | Evaluation return separated from training/shaped return |
| Episode length, time to win/defeat | Advanced | Completion-aware duration; conditional times require outcome exposure/censoring |
| Comebacks, recovered deficits, surrendered leads, lead changes, first-score effects, close/decisive wins | Advanced descriptive | Score-trajectory slices; post-treatment and gameable, never primary |
| Individual kills, last hits, agent K/D, general elimination participation, solo kills, pentakills | Reject | Lethal-transition damage-source truth, single-/multi-contributor lethal transitions, team wipes |
| Damage dealt/taken | Retain | Recipient-modified gross stages with exact units and shares |
| “Effective healing,” overheal | Correct and retain selectively | Gross healing primary; recipient-level net/clamp outcomes advanced; no arbitrary source realization |
| Damage/healing per engagement or teamfight | Reject | No generic engagement/teamfight segmentation is planned; use authoritative task context or controlled scenarios for a named question |
| Damage by target class/status/ability/context | Retain as dimensions | Long-form slices of one amount metric |
| Focus-fire events/concentration | Conditional key secondary | Same-transition eligible multi-attacker concentration plus exposure |
| Contributors per death, multi-agent deaths | Retain with narrowed meaning | Lethal-transition source-count distribution and multi-contributor lethal-transition rate |
| Teamfight W/D/L, participation, damage, healing, CC, numerical advantage | Reject | No generic teamfight detector is planned and no universal strategic win definition exists |
| Damage mitigation/amplification | Retain combined value | Exact team/class combined effect; duplicate emitter credit blocked |
| Survival rate or deaths per engagement/teamfight | Reject | Depends on the rejected generic segmentation and encourages unsupported quality claims |
| Time alive/dead, life duration, respawn time, wave size | Advanced diagnostics | Authoritative lifecycle quantities, not compact tactical-quality endpoints |
| Critical-health escapes | Scenario/advanced association | Low-health state alone does not prove skill |
| Lethal/clutch saves | Retain corrected definition | Same-transition lethal-damage rescue; critical-health survival is a separate association |
| Healing by recipient class, share, received | Retain as dimensions | Gross-healing long-form slices |
| Ally deaths in heal radius / while Ultimate ready | Reject as headline | Exposure and counterfactual quality are undefined; scenario if needed |
| CC duration/uptime/applications/lifecycle | Retain advanced | Channel-specific; no universal CC score |
| Damage/death under CC | Retain advanced association | Do not claim causal combo quality |
| Trap casts/triggers/placed uptime | Correct | Targeted applications, active steps, lifecycle episode endings, damage-break rate |
| Anti-heal uptime/healing prevented | Retain combined value | Preserve healing exposure; no per-Rogue credit under overlap |
| Ultimate use/conversion/kill within K | Activations retained; conversion conditional | Mechanic-native active-window association only |
| Burst uptime/damage/deaths/excess | Retain active-window amounts; reject excess counterfactual | No hypothetical attacks or “kills caused” wording |
| Freedom uptime/effective time/movement recovered | Retain binding coverage; scenario for utility | No counterfactual distance fact |
| Warrior tanking, Hunter low damage taken | Retain descriptive exposure | Not proof of tanking quality or kiting |
| Peeling, kiting, flanking, body blocking, backline access, cornering, LOS/choke use, escape denial | Scenario | Context-dependent geometry behavior |
| Regeneration/recovery | Retain advanced diagnostic | Separate from Priest healing; mobile class-specific mechanic |
| KoTH control/contest/score/captures/defenses/allocations | Future task owner | Eligible hill-timestep denominators; no per-agent score apportionment |
| Objective fight cost deaths/score | Reject | Unstable at low/zero score and strategically ambiguous |
| CTF pickups/captures/drops/returns/possession/interceptions | Future task owner | Exact task-authored edges and opportunity denominators |
| Preferred carrier quality | Reject | Carry-by-class is descriptive; do not encode designer preference |
| Escort/bodyguard/kill-squad size | Advanced descriptive | Scenario owns decision quality |
| Local numerical advantage, cohesion, pairwise distance, centroids | Advanced descriptive/scenario | Not universal quality and no new core fact |
| Action acceptance/rejection | Diagnostic | Upstream policy/sampler and legality quality control |
| Temporal slices while ahead/behind/contesting | Retain as dimensions | Do not create primitive metric explosion |
| Cross-play gap, partner robustness, worst partner, ratings | Retain with corrections | Frozen pools/full matrix; lower-tail not raw minimum; ratings secondary |
| Sample efficiency and runtime | Retain protocol-owned | Fixed held-out evaluations and declared hardware/JIT protocol |
| Opaque tactical/coordination score | Reject | Conflicts with interpretability and hides behavior tradeoffs |

## Verification requirements

Before a metric becomes active, its owner must provide:

- hand-constructed neutral, positive, negative, and zero-opportunity traces;
- team-swap and side-swap invariance where applicable;
- canonical SharedObs mode/projection checks and exact configured-roster
  availability equality on every official replay frame, plus generic
  NoSharedObs and permitted-subset SharedObs compatibility tests;
- complete/partial/interrupted/failed rollout tests plus independent observed,
  right-censored, competing-event, unavailable, and not-applicable endpoint
  tests where relevant;
- simultaneous damage/healing/clamp and duplicate-class overlap cases;
- live-model and replay-loaded parity once file replay exists;
- raw sufficient-statistic and presentation-roundtrip checks;
- adversarial gaming/counterexample review;
- construct validation for any coordination or scenario interpretation;
- matched full-method/ablation scenario evidence with independently trained
  pairs, one predeclared primary endpoint, and proof that no scenario content or
  result entered training, selection, shaping, or curriculum decisions;
- for a Big 12 rating, exact twelve-system cardinality, complete 6,600-episode
  outcome/failure accounting, validation-only final-system selection, immutable
  roster-snapshot identity, and a qualified draw-aware estimator; and
- explicit Four-North-Star verdicts.

No unresolved `blocked` or `validation_pending` row may appear as an official
paper endpoint, benchmark score, reward preset, or release claim.
