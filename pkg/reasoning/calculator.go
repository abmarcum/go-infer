package reasoning

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

// TokenType represents the lexical token type in a math expression.
type TokenType int

const (
	TokEOF TokenType = iota
	TokNumber
	TokIdent
	TokPlus
	TokMinus
	TokMul
	TokDiv
	TokMod
	TokPow
	TokFact
	TokLParen
	TokRParen
	TokComma
)

// Token represents a lexical token in a math expression.
type Token struct {
	Type  TokenType
	Text  string
	Value float64
}

// Lexer breaks a mathematical expression into tokens.
type Lexer struct {
	input []rune
	pos   int
}

// NewLexer creates a math lexer.
func NewLexer(input string) *Lexer {
	return &Lexer{input: []rune(input), pos: 0}
}

func (l *Lexer) peek() rune {
	if l.pos >= len(l.input) {
		return 0
	}
	return l.input[l.pos]
}

func (l *Lexer) next() rune {
	if l.pos >= len(l.input) {
		return 0
	}
	r := l.input[l.pos]
	l.pos++
	return r
}

func (l *Lexer) skipWhitespace() {
	for l.pos < len(l.input) && unicode.IsSpace(l.input[l.pos]) {
		l.pos++
	}
}

// NextToken scans the next token.
func (l *Lexer) NextToken() (Token, error) {
	l.skipWhitespace()
	if l.pos >= len(l.input) {
		return Token{Type: TokEOF}, nil
	}

	r := l.peek()

	// Numbers
	if unicode.IsDigit(r) || (r == '.' && l.pos+1 < len(l.input) && unicode.IsDigit(l.input[l.pos+1])) {
		start := l.pos
		hasDot := false
		for l.pos < len(l.input) {
			ch := l.input[l.pos]
			if ch == '.' {
				if hasDot {
					break
				}
				hasDot = true
				l.pos++
			} else if unicode.IsDigit(ch) {
				l.pos++
			} else if (ch == 'e' || ch == 'E') && l.pos+1 < len(l.input) {
				// Handle scientific notation e.g., 1e-4, 2E5
				l.pos++
				if l.pos < len(l.input) && (l.input[l.pos] == '+' || l.input[l.pos] == '-') {
					l.pos++
				}
			} else {
				break
			}
		}
		txt := string(l.input[start:l.pos])
		val, err := strconv.ParseFloat(txt, 64)
		if err != nil {
			return Token{}, fmt.Errorf("invalid number '%s': %w", txt, err)
		}
		return Token{Type: TokNumber, Text: txt, Value: val}, nil
	}

	// Identifiers (functions, constants)
	if unicode.IsLetter(r) {
		start := l.pos
		for l.pos < len(l.input) && (unicode.IsLetter(l.input[l.pos]) || unicode.IsDigit(l.input[l.pos]) || l.input[l.pos] == '_') {
			l.pos++
		}
		txt := string(l.input[start:l.pos])
		return Token{Type: TokIdent, Text: strings.ToLower(txt)}, nil
	}

	l.next()
	switch r {
	case '+':
		return Token{Type: TokPlus, Text: "+"}, nil
	case '-':
		return Token{Type: TokMinus, Text: "-"}, nil
	case '*':
		if l.peek() == '*' { // Python style exponentiation **
			l.next()
			return Token{Type: TokPow, Text: "**"}, nil
		}
		return Token{Type: TokMul, Text: "*"}, nil
	case '/':
		return Token{Type: TokDiv, Text: "/"}, nil
	case '%':
		return Token{Type: TokMod, Text: "%"}, nil
	case '^':
		return Token{Type: TokPow, Text: "^"}, nil
	case '!':
		return Token{Type: TokFact, Text: "!"}, nil
	case '(':
		return Token{Type: TokLParen, Text: "("}, nil
	case ')':
		return Token{Type: TokRParen, Text: ")"}, nil
	case ',':
		return Token{Type: TokComma, Text: ","}, nil
	default:
		return Token{}, fmt.Errorf("unexpected character '%c'", r)
	}
}

// Parser performs recursive descent parsing of mathematical expressions.
type Parser struct {
	lexer *Lexer
	cur   Token
}

// NewParser creates a math parser.
func NewParser(input string) (*Parser, error) {
	p := &Parser{lexer: NewLexer(input)}
	var err error
	p.cur, err = p.lexer.NextToken()
	if err != nil {
		return nil, err
	}
	return p, nil
}

func (p *Parser) consume(expected TokenType) error {
	if p.cur.Type != expected {
		return fmt.Errorf("syntax error: expected token %v, got %v (%s)", expected, p.cur.Type, p.cur.Text)
	}
	var err error
	p.cur, err = p.lexer.NextToken()
	return err
}

// Parse parses and evaluates the expression.
func (p *Parser) Parse() (float64, error) {
	val, err := p.parseExpr()
	if err != nil {
		return 0, err
	}
	if p.cur.Type != TokEOF {
		return 0, fmt.Errorf("unexpected extra token at end: %s", p.cur.Text)
	}
	return val, nil
}

// parseExpr handles addition and subtraction: + -
func (p *Parser) parseExpr() (float64, error) {
	val, err := p.parseTerm()
	if err != nil {
		return 0, err
	}

	for p.cur.Type == TokPlus || p.cur.Type == TokMinus {
		op := p.cur.Type
		if err := p.consume(op); err != nil {
			return 0, err
		}
		right, err := p.parseTerm()
		if err != nil {
			return 0, err
		}
		if op == TokPlus {
			val += right
		} else {
			val -= right
		}
	}
	return val, nil
}

// parseTerm handles multiplication, division, modulo: * / %
func (p *Parser) parseTerm() (float64, error) {
	val, err := p.parsePower()
	if err != nil {
		return 0, err
	}

	for p.cur.Type == TokMul || p.cur.Type == TokDiv || p.cur.Type == TokMod {
		op := p.cur.Type
		if err := p.consume(op); err != nil {
			return 0, err
		}
		right, err := p.parsePower()
		if err != nil {
			return 0, err
		}
		switch op {
		case TokMul:
			val *= right
		case TokDiv:
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			val /= right
		case TokMod:
			if right == 0 {
				return 0, fmt.Errorf("modulo by zero")
			}
			val = math.Mod(val, right)
		}
	}
	return val, nil
}

// parsePower handles exponentiation: ^ or ** (right associative)
func (p *Parser) parsePower() (float64, error) {
	val, err := p.parseUnary()
	if err != nil {
		return 0, err
	}

	if p.cur.Type == TokPow {
		if err := p.consume(TokPow); err != nil {
			return 0, err
		}
		right, err := p.parsePower() // right-associative
		if err != nil {
			return 0, err
		}
		val = math.Pow(val, right)
	}
	return val, nil
}

// parseUnary handles unary minus and plus: -x, +x
func (p *Parser) parseUnary() (float64, error) {
	if p.cur.Type == TokMinus {
		if err := p.consume(TokMinus); err != nil {
			return 0, err
		}
		val, err := p.parseUnary()
		if err != nil {
			return 0, err
		}
		return -val, nil
	}
	if p.cur.Type == TokPlus {
		if err := p.consume(TokPlus); err != nil {
			return 0, err
		}
		return p.parseUnary()
	}
	return p.parsePostfix()
}

// parsePostfix handles postfix factorial: x!
func (p *Parser) parsePostfix() (float64, error) {
	val, err := p.parseFactor()
	if err != nil {
		return 0, err
	}

	for p.cur.Type == TokFact {
		if err := p.consume(TokFact); err != nil {
			return 0, err
		}
		if val < 0 || math.Floor(val) != val || val > 170 {
			return 0, fmt.Errorf("factorial requires non-negative integer <= 170, got %g", val)
		}
		val = factorial(int(val))
	}
	return val, nil
}

// parseFactor handles numbers, identifiers/functions, constants, and parenthesized expressions
func (p *Parser) parseFactor() (float64, error) {
	switch p.cur.Type {
	case TokNumber:
		val := p.cur.Value
		if err := p.consume(TokNumber); err != nil {
			return 0, err
		}
		return val, nil

	case TokIdent:
		name := p.cur.Text
		if err := p.consume(TokIdent); err != nil {
			return 0, err
		}

		// Constants
		switch name {
		case "pi":
			return math.Pi, nil
		case "e":
			return math.E, nil
		}

		// Functions with parentheses
		if p.cur.Type != TokLParen {
			return 0, fmt.Errorf("unknown identifier or missing parentheses after function '%s'", name)
		}
		if err := p.consume(TokLParen); err != nil {
			return 0, err
		}

		args, err := p.parseArgList()
		if err != nil {
			return 0, err
		}
		if err := p.consume(TokRParen); err != nil {
			return 0, err
		}

		return evaluateFunc(name, args)

	case TokLParen:
		if err := p.consume(TokLParen); err != nil {
			return 0, err
		}
		val, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		if err := p.consume(TokRParen); err != nil {
			return 0, err
		}
		return val, nil

	default:
		return 0, fmt.Errorf("unexpected token in expression: %s", p.cur.Text)
	}
}

func (p *Parser) parseArgList() ([]float64, error) {
	var args []float64
	if p.cur.Type == TokRParen {
		return args, nil
	}
	arg, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	args = append(args, arg)

	for p.cur.Type == TokComma {
		if err := p.consume(TokComma); err != nil {
			return nil, err
		}
		nextArg, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		args = append(args, nextArg)
	}
	return args, nil
}

func evaluateFunc(name string, args []float64) (float64, error) {
	switch name {
	case "sqrt":
		if len(args) != 1 {
			return 0, fmt.Errorf("sqrt requires 1 argument, got %d", len(args))
		}
		if args[0] < 0 {
			return 0, fmt.Errorf("sqrt of negative number %g", args[0])
		}
		return math.Sqrt(args[0]), nil

	case "cbrt":
		if len(args) != 1 {
			return 0, fmt.Errorf("cbrt requires 1 argument, got %d", len(args))
		}
		return math.Cbrt(args[0]), nil

	case "abs":
		if len(args) != 1 {
			return 0, fmt.Errorf("abs requires 1 argument, got %d", len(args))
		}
		return math.Abs(args[0]), nil

	case "sin":
		if len(args) != 1 {
			return 0, fmt.Errorf("sin requires 1 argument, got %d", len(args))
		}
		return math.Sin(args[0]), nil

	case "cos":
		if len(args) != 1 {
			return 0, fmt.Errorf("cos requires 1 argument, got %d", len(args))
		}
		return math.Cos(args[0]), nil

	case "tan":
		if len(args) != 1 {
			return 0, fmt.Errorf("tan requires 1 argument, got %d", len(args))
		}
		return math.Tan(args[0]), nil

	case "log", "log10":
		if len(args) != 1 {
			return 0, fmt.Errorf("log requires 1 argument, got %d", len(args))
		}
		if args[0] <= 0 {
			return 0, fmt.Errorf("log of non-positive number %g", args[0])
		}
		return math.Log10(args[0]), nil

	case "ln":
		if len(args) != 1 {
			return 0, fmt.Errorf("ln requires 1 argument, got %d", len(args))
		}
		if args[0] <= 0 {
			return 0, fmt.Errorf("ln of non-positive number %g", args[0])
		}
		return math.Log(args[0]), nil

	case "exp":
		if len(args) != 1 {
			return 0, fmt.Errorf("exp requires 1 argument, got %d", len(args))
		}
		return math.Exp(args[0]), nil

	case "floor":
		if len(args) != 1 {
			return 0, fmt.Errorf("floor requires 1 argument, got %d", len(args))
		}
		return math.Floor(args[0]), nil

	case "ceil":
		if len(args) != 1 {
			return 0, fmt.Errorf("ceil requires 1 argument, got %d", len(args))
		}
		return math.Ceil(args[0]), nil

	case "round":
		if len(args) != 1 {
			return 0, fmt.Errorf("round requires 1 argument, got %d", len(args))
		}
		return math.Round(args[0]), nil

	case "pow":
		if len(args) != 2 {
			return 0, fmt.Errorf("pow requires 2 arguments, got %d", len(args))
		}
		return math.Pow(args[0], args[1]), nil

	case "max":
		if len(args) < 1 {
			return 0, fmt.Errorf("max requires at least 1 argument")
		}
		m := args[0]
		for _, a := range args[1:] {
			if a > m {
				m = a
			}
		}
		return m, nil

	case "min":
		if len(args) < 1 {
			return 0, fmt.Errorf("min requires at least 1 argument")
		}
		m := args[0]
		for _, a := range args[1:] {
			if a < m {
				m = a
			}
		}
		return m, nil

	default:
		return 0, fmt.Errorf("unknown function '%s'", name)
	}
}

func factorial(n int) float64 {
	res := 1.0
	for i := 2; i <= n; i++ {
		res *= float64(i)
	}
	return res
}

// FormatResult formats a float64 into a clean human-readable representation.
func FormatResult(val float64) string {
	if math.IsNaN(val) {
		return "NaN"
	}
	if math.IsInf(val, 1) {
		return "Infinity"
	}
	if math.IsInf(val, -1) {
		return "-Infinity"
	}

	// Round to nearest if very close to integer to avoid IEEE 754 float drift (e.g., 0.1 + 0.2)
	rounded := math.Round(val)
	if math.Abs(val-rounded) < 1e-12 {
		return fmt.Sprintf("%.0f", rounded)
	}

	return strconv.FormatFloat(val, 'g', 10, 64)
}

// EvaluateMath takes an expression string and evaluates it, returning the float value and formatted string.
func EvaluateMath(expr string) (float64, string, error) {
	clean := strings.TrimSpace(expr)
	if clean == "" {
		return 0, "", fmt.Errorf("empty expression")
	}

	// Remove common wrapper prefixes/suffixes if present
	clean = strings.TrimPrefix(clean, "=")
	clean = strings.TrimSpace(clean)

	parser, err := NewParser(clean)
	if err != nil {
		return 0, "", err
	}

	val, err := parser.Parse()
	if err != nil {
		return 0, "", err
	}

	return val, FormatResult(val), nil
}
