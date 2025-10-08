%{
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

int yylex(void);
int yyerror(const char *s);
%}

%union {
    double dval;
}

%token <dval> NUMBER
%type <dval> expr
%left '+' '-'
%left '*' '/'
%right UMINUS

%%

lines:
    /* empty */
  | lines expr '\n'   { printf("= %g\n", $2); }
  | lines '\n'
  ;

expr:
    expr '+' expr      { $$ = $1 + $3; }
  | expr '-' expr      { $$ = $1 - $3; }
  | expr '*' expr      { $$ = $1 * $3; }
  | expr '/' expr      { $$ = $1 / $3; }
  | '-' expr %prec UMINUS { $$ = -$2; }
  | '(' expr ')'       { $$ = $2; }
  | NUMBER             { $$ = $1; }
  ;

%%

int yylex(void) {
    int c;
    while ((c = getchar()) == ' ' || c == '\t');  // skip spaces

    if (isdigit(c) || c == '.') {
        ungetc(c, stdin);
        yylval.dval = 0;
        scanf("%lf", &yylval.dval);
        return NUMBER;
    }

    return c;
}

int yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
    return 0;
}

int main(void) {
    printf("Enter expressions (Ctrl+D to exit):\n");
    yyparse();
    return 0;
}
