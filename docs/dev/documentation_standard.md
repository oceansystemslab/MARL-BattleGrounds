# Documentation Standard

**EASE. OF. USE.** A reader should be able to use a function correctly from its
signature and editor hover text. Use simple, direct English. Say who does what.
Explain needed technical words, and keep every fact needed for correct use.
Short wording must not hide limits, defaults or side effects.

This is the shared standard for all existing and new project documentation:
Python, JavaScript, shell tools, examples, guides and in-file comments. It covers
public interfaces, private helpers, classes, records and source files. Accurate
documentation is part of each change's acceptance criteria, not later polish.

## Test Files

**Each test file has exactly one file-level docstring.** Explain its purpose and
the contracts it checks at the top. Do not add class, function, method or fixture
docstrings. Remove existing ones when applying this standard. Keep assertions,
fixtures, names, parametrization and behavior unchanged. This exception also
applies to test-only fixture, harness, support and helper modules, even when
their filenames do not start with `test`. Production and tool helper modules
still require complete callable documentation.

For JavaScript tests, use one file-level `@file` comment. Keep JSDoc type
annotations required by the type checker. Necessary type directives are not
function prose. Short comments may explain a non-obvious assertion, but do not
replace removed docstrings with repetitive comments.

## Plans and Implementation Packets

Every new milestone plan and implementation packet must have a
`Documentation Contract` section linking here. Active packets must have one too.
Name the files and interfaces to document, pair documentation review with each
code checkpoint, and include accurate file descriptions, hover text and examples
in its done criteria. Apply this standard rather than copying a new version into
each packet. Private working notes must remain private; public guides must not
depend on them.

## Write the Contract

- Use NumPy-style Python docstrings. Use a short summary, then the sections
  needed by the actual contract: `Parameters`, `Returns` or `Yields`, `Raises`,
  `Attributes`, `Notes`, `Examples` and `See Also`. Use JSDoc for JavaScript and
  TypeScript. Use a purpose and usage comment at the top of shell scripts.
  Keep the same content rules across languages. Do not add empty sections.
- Describe every source file at its top. Say what it owns, who uses it, its
  main entry points and any important boundary. For a script, explain how to
  run it, what it reads or writes, and its prerequisites. For a test file,
  explain which contract it checks. Keep shebangs, encoding lines, legal
  notices and required directive ordering valid.
- Start each callable's documentation with what it does for its caller. Explain
  each argument's meaning, accepted values, units and default behavior. Include
  positional and keyword-only arguments, optional values, and precedence when
  several inputs can supply the same setting. Do not repeat `self` or `cls`.
- For arrays, state shape, axis meaning, dtype, scalar/batch behavior and relevant
  ranges or masks. Say whether IDs are global, team-local, actor-relative or
  external. State the decision or transition time that a value describes.
- Explain every return value, its order and meaning. Distinguish `None`, an
  empty result, a missing measurement, zero and an inactive/padded entry. Explain
  who owns returned state, whether inputs are changed and what must be carried
  into the next call.
- Document errors callers can reasonably encounter and the conditions that
  trigger them. Separate checks the function performs from preconditions the
  caller must satisfy. Do not promise validation, recovery or compatibility
  that the implementation does not provide.
- State relevant file/network/device effects, resource ownership, randomness,
  reset/terminal behavior, ordering and thread/process assumptions. For JAX,
  explain host-only setup, compiled use, static structure and dynamic values
  where these affect correct use. Do not claim that all shapes share one
  compiled program or that matching seeds guarantee every external method's
  actions.
- Document classes and records as contracts. Describe meaningful fields and
  their defaults, mutability and invariants. Keep field documentation beside
  the field when the editor can show it there. Do not merely expand an acronym
  or repeat the field name.
- Use small complete examples when they make correct use clearer. Include
  imports, required setup and distinct random keys where needed. Use
  `import marl_battlegrounds as marl_bgs` in researcher examples. Clearly label
  fragments and proposed APIs. An example must not silently rely on private
  milestone notes, hidden variables or a future feature.
- Scale length to the real contract. A small private helper may need only a
  precise paragraph; a public reset or evaluator needs more. Explain the
  reasons for non-obvious code in comments. Avoid filler, line-by-line narration,
  generic generated summaries and copied contracts that will drift apart.
  Give each rule one owner and link to it where more detail is useful.

## Check the Contract

- Read the implementation, its called authorities and representative callers
  before writing a contract. Check defaults and edge cases against source and
  tests. Do not infer behavior from a name, an annotation or an older proposal.
  Resolve conflicting evidence; otherwise state the exact uncertainty.
- Review existing documentation as carefully as missing documentation. Correct
  stale claims, ambiguous ownership, wrong units, incomplete defaults and
  examples that no longer match the supported interface.
- Keep current behavior, accepted future design and historical evidence clearly
  separate. Preserve audit hashes, quoted source text, historical schemas and
  immutable evidence. Clarify their status around them rather than rewriting
  what they originally recorded. Do not edit vendored or generated output as
  though it were the source authority.
- For a documentation-only change, prove executable code is unchanged. Compare
  Python syntax trees with documentation strings removed, or JavaScript tokens
  without comments, as appropriate. Also inspect reflection: docstrings can
  affect help output, command help, generated schemas and documentation exports.
  Syntax-tree equality alone does not prove those outputs are unchanged.
- Check formatting, links, documented signatures and example syntax. Run useful
  examples or reuse matching execution evidence. Have another reviewer inspect
  material public contracts against fresh source. A count of docstrings is
  coverage evidence, not proof of correctness or clarity.
- Record scope, checks and remaining gaps honestly. Do not claim the whole
  repository meets this standard after checking only a few public functions.
