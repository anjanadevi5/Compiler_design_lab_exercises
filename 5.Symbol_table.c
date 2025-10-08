#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int main() {
    char expr[50];
    char symbol[20];
    void *address[20];
    int i = 0, x = 0;

    printf("Input the expression ending with $ sign: ");

    // Read expression until '$'
    char c;
    while ((c = getchar()) != '$' && i < 49) {
        expr[i++] = c;
    }
    expr[i] = '\0'; // terminate string

    printf("\nGiven Expression: %s\n", expr);
    printf("\nSymbol Table Display\n");
    printf("Symbol\tAddress\t\tType\n");

    // Process each character
    for (i = 0; expr[i] != '\0'; i++) {
        if (isalpha(expr[i])) {
            // Identifier
            address[x] = malloc(sizeof(char));
            symbol[x] = expr[i];
            printf("%c\t%p\tidentifier\n", expr[i], address[x]);
            x++;
        } 
        else if (isdigit(expr[i])) {
            // Constant
            address[x] = malloc(sizeof(int));
            symbol[x] = expr[i];
            printf("%c\t%p\tconstant\n", expr[i], address[x]);
            x++;
        }
        else if (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '=') {
            // Operator
            address[x] = malloc(sizeof(char));
            symbol[x] = expr[i];
            printf("%c\t%p\toperator\n", expr[i], address[x]);
            x++;
        }
    }

    return 0;
}
