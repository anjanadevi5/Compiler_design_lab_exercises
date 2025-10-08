#include <stdio.h>
#include <string.h>

int main() {
    char exp[50];
    int temp = 1; // use integer for temp counter

    printf("Enter an expression (like a=b+c*d): ");
    scanf("%s", exp);

    printf("\nIntermediate Code:\n");

    // Process * and / first (higher precedence)
    for (int i = 0; exp[i] != '\0'; i++) {
        if (exp[i] == '*' || exp[i] == '/') {
            printf("t%d = %c %c %c\n", temp, exp[i - 1], exp[i], exp[i + 1]);
            exp[i - 1] = 'A' + temp - 1; // placeholder (A, B, C...) for t1, t2
            memmove(&exp[i], &exp[i + 2], strlen(exp) - i - 1);
            temp++;
            i = -1;
        }
    }

    // Process + and - next
    for (int i = 0; exp[i] != '\0'; i++) {
        if (exp[i] == '+' || exp[i] == '-') {
            printf("t%d = %c %c %c\n", temp, exp[i - 1], exp[i], exp[i + 1]);
            exp[i - 1] = 'A' + temp - 1;
            memmove(&exp[i], &exp[i + 2], strlen(exp) - i - 1);
            temp++;
            i = -1;
        }
    }

    // Final assignment
    printf("%c = t%d\n", exp[0], temp - 1);

    return 0;
}
