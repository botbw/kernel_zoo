#include <bits/stdc++.h>
using namespace std;

using uint8 = unsigned char;

constexpr int K = 4;

struct INT
{
    uint8 data[K];

    INT()
    {
        for (int i = 0; i < K; ++i)
        {
            data[i] = 0;
        }
    }

    explicit INT(uint32_t value)
    {
        for (int i = 0; i < K; ++i)
        {
            data[i] = (value >> (8 * i)) & 0xFF;
        }
    }

    explicit INT(uint64_t value)
    {
        for (int i = 0; i < K; ++i)
        {
            data[i] = (value >> (8 * i)) & 0xFF;
        }
    }

    uint8 &operator[](size_t index)
    {
        if (index >= K)
        {
            throw out_of_range("Index out of range");
        }
        return data[index];
    }

    const uint8 &operator[](size_t index) const
    {
        if (index >= K)
        {
            throw out_of_range("Index out of range");
        }
        return data[index];
    }

    uint32_t toUInt32() const
    {
        uint32_t value = 0;
        for (int i = 0; i < K; ++i)
        {
            value |= static_cast<uint32_t>(data[i]) << (8 * i);
        }
        return value;
    }

    void printBytes() const
    {
        for (int i = 3; i >= 0; --i)
        {
            cout << hex << setw(2) << setfill('0') << static_cast<int>(data[i]) << " ";
        }
        cout << dec << endl;
    }

    INT &operator=(uint32_t value)
    {
        for (int i = 0; i < K; ++i)
        {
            data[i] = (value >> (8 * i)) & 0xFF;
        }
        return *this;
    }
};

void generateRandomMatrix_(INT matrix[16][16], uint32_t minValue = 0, uint32_t maxValue = 100)
{
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            matrix[i][j] = minValue + (rand() % (maxValue - minValue + 1));
        }
    }
}

void printMatrix(INT matrix[16][16])
{
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            cout << setw(4) << matrix[i][j].toUInt32() << " ";
        }
        cout << endl;
    }
}

void cudaCoreMatmul_(INT A[16][16], INT B[16][16], INT C[16][16])
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                C[i][j] = A[i][k].toUInt32() * B[k][j].toUInt32() + C[i][j].toUInt32();
            }
        }
    }
}

void mem_format_transform_(uint8 out[K][16][16], INT in[16][16])
{
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                out[i][j][k] = in[j][k][i];
            }
        }
    }
}

void elementwise_add_(INT out[16][16], uint32_t in[16][16])
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            out[i][j] = in[i][j] + out[i][j].toUInt32();
        }
    }
}

void mma_m16n16k16_(uint8 A[16][16], uint8 B[16][16], uint32_t C[16][16], uint32_t alpha, uint32_t beta)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                C[i][j] = A[i][k] * B[k][j] + C[i][j];
            }
            C[i][j] = alpha * C[i][j] + beta;
        }
    }
}

void tensorCoreMatmul_(INT A[16][16], INT B[16][16], INT C[16][16])
{
    uint8 A_formatted[K][16][16], B_formatted[K][16][16];
    mem_format_transform_(A_formatted, A);
    mem_format_transform_(B_formatted, B);

    for (int x = 0; x < K; x++)
    {
        for (int y = 0; y < K; y++)
        {
            uint32_t C_partial[16][16] = {0};
            mma_m16n16k16_(A_formatted[x], B_formatted[y], C_partial, 1 << 8 * (x + y), 0);
            elementwise_add_(C, C_partial);
        }
    }
}

int main()
{
    srand(time(0));
    INT A[16][16], B[16][16], C[16][16], C_ref[16][16];
    generateRandomMatrix_(A);
    generateRandomMatrix_(B);
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            C[i][j] = C_ref[i][j] = 0;
        }
    }
    cudaCoreMatmul_(A, B, C_ref);
    tensorCoreMatmul_(A, B, C);
    puts("C_ref:\n");
    printMatrix(C_ref);
    puts("\n\nC:\n");
    printMatrix(C);
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            assert(C_ref[i][j].toUInt32() == C[i][j].toUInt32());
        }
    }
    return 0;
}
