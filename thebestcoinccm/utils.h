#pragma once


template <size_t SIZE>
void dump(const char * title, uint8_t (&arr)[SIZE])
{
    printf("%s", title);
    for (size_t i = 0; i < SIZE; i++)
        printf(" %02X", unsigned(arr[i]));
    printf("\n");
}
