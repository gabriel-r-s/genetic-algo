#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <SDL2/SDL.h>

//========================================================================
// Funções para codificação binária e auxiliares de matemática 
// `Num` representa um número com 4 bits de parte inteira,
// 27 bits fracionários, e 1 bit de sinal
//========================================================================

// núm. de bits fracionários
#define SIZE 32
#define FRAC 27


// Gera um double aleatorio em [lo, hi]
double randd(double lo, double hi) {
    double rand_01 = (double) rand() / (double) RAND_MAX;
    return rand_01 * (hi - lo) + lo;
}

// gera um num. de 32 bits, onde cada bit tem apenas rate% de chance de ser 1
// para uso de mutação de número, uso: gene ^= mut32(tx_mut)
uint32_t mut32(double rate) {
    uint32_t mut = 0;
    for (int i = 0; i < SIZE; ++i) {
        mut <<= 1;
        if (randd(0, 1) <= rate)
            mut |= 1;
    }
    return mut;
}


// converte entre unidades, x em [a1, a2], f(x) em [b1, b2]
double unit_map(double x, double a1, double a2, double b1, double b2) {
    return (x - a1) / (a2 - a1) * (b2 - b1) + b1;
}


//========================================================================


typedef union {
    struct {
        unsigned frac: FRAC;
        unsigned whol: (SIZE - FRAC - 1);
        unsigned sign: 1; 
    } parts;
    uint32_t bits;
} Num;


// limita o valor absoluto de n para dentro de [-10, 10]
Num num_clamp(Num n) {
    if (n.parts.whol >= 10) { 
        n.parts.whol = 10;
        n.parts.frac = 0;
    }
    return n;
}


// constroi um Num aleatório em [-10, 10]
Num num_rand() {
    Num n = { .bits =
          (rand() & 0xFF)
        | (rand() & 0xFF) << 8
        | (rand() & 0xFF) << 16
        | (rand() & 0xFF) << 24
    };
    return num_clamp(n);
}


// converte o valor de ponto fixo para double
double num_to_double(Num n) {
    double x = ((double) (n.parts.whol << FRAC | n.parts.frac)) * pow(2.0, -FRAC);
    if (n.parts.sign) x = -x;
    return x;
}


// A função avaliada
double f(double x, double y) {
    return
          0.97*exp(-0.2*((x+3)*(x+3) + (y+3)*(y+3)))
        + 0.98*exp(-0.2*((x+3)*(x+3) + (y-3)*(y-3)))
        + 0.99*exp(-0.2*((x-3)*(x-3) + (y+3)*(y+3)))
        + 1.00*exp(-0.2*((x-3)*(x-3) + (y-3)*(y-3)));
}


//========================================================================
// Funções de algoritmo genético, indivíduos, gerações
//========================================================================

// Estrutura representando um indivíduo
typedef struct {
    Num x, y;
    double fit;
} Ind;

// Cruzamento binário de indivíduos
void ind_mix(Ind p1, Ind p2, uint32_t mask, Ind *f1, Ind *f2) {
    f1->x.bits = (p1.x.bits & ~mask) | (p2.x.bits & mask);
    f1->y.bits = (p1.y.bits & ~mask) | (p2.y.bits & mask);

    f2->x.bits = (p1.x.bits & mask) | (p2.x.bits & ~mask);
    f2->y.bits = (p1.y.bits & mask) | (p2.y.bits & ~mask);
}

// Mutação binária de indivíduos
Ind ind_mutate(Ind ind, double rate) {
    ind.x.bits ^= mut32(rate);
    ind.x = num_clamp(ind.x);

    ind.y.bits ^= mut32(rate);
    ind.y = num_clamp(ind.y);

    return ind;
}

// Função aux. para qsort de invivíduos (descendente)
int ind_sort_rev(const void *v_a, const void *v_b) {
    const Ind *a = (Ind *)v_a;
    const Ind *b = (Ind *)v_b;

    if (a->fit > b->fit)
        return -1;
    else if (a->fit == b->fit)
        return 0;
    else if (a->fit < b->fit)
        return 1;
    else {
        fprintf(stderr, "Problema de ponto de flutuante!\n");
        exit(1);
    }
}


// Estrutura representando os parâmetros de algoritmo genético,
// variáveis, e alocações do estado atual do algoritmo
typedef struct {
    size_t gen, pop_size, sel_size, elite_size;
    uint32_t cruz_mask;
    double tx_cruz, tx_mut;
    size_t *pop_sel_idx;
    Ind *pop0, *pop1, *pop_sel;
} Ag;


// Libera os buffers alocados
void ag_free(Ag ag) {
    free(ag.pop0);
    free(ag.pop1);
    free(ag.pop_sel_idx);
    free(ag.pop_sel);
}


// Inicialização do algoritmo, de acordo com os parâmetros
Ag ag_init(size_t pop_size, size_t sel_size, double elite, double tx_cruz, double tx_mut, uint32_t cruz_mask) {
    // evitando números ímpares, arredondando pra cima
    pop_size += pop_size % 2;
    size_t elite_size = (double) pop_size * elite;
    elite_size += elite_size % 2;

    size_t *pop_sel_idx = malloc(sel_size*sizeof(*pop_sel_idx));
    Ind *pop0 = malloc(pop_size*sizeof(*pop0));
    Ind *pop1 = malloc(pop_size*sizeof(*pop1));
    Ind *pop_sel = malloc(sel_size*sizeof(*pop_sel));

    for (size_t i = 0; i < pop_size; ++i) {
        Num x = num_rand();
        Num y = num_rand();
        double fit = f(num_to_double(x), num_to_double(y));

        pop0[i] = (Ind) { .x = x, .y = y, .fit = fit };
    }
    qsort(pop0, pop_size, sizeof(Ind), ind_sort_rev);


    return (Ag) {
        .gen = 0, .pop_size = pop_size, .sel_size = sel_size, .elite_size = elite_size,
        .cruz_mask = cruz_mask,
        .tx_cruz = tx_cruz, .tx_mut = tx_mut,
        .pop_sel_idx = pop_sel_idx,
        .pop0 = pop0, .pop1 = pop1, .pop_sel = pop_sel,
    };
}


// Função aux. para selecionar `sel_size` indivíduos distintos
// obs: ao inves iterar 2 vezes para gerar 2 pais 
void ag_select(Ag *ag) {
    for (size_t i = 0; i < ag->sel_size; ++i) {
        // seleciona um indice aleatorio
        // repete enquanto indice ja estiver presente nos indices da seleção
        size_t idx;
        bool already_sel;
        do {
            idx = rand() % ag->pop_size;
            already_sel = false;

            for (size_t j = 0; j < i; ++j) {
                if (ag->pop_sel_idx[j] == idx) {
                    already_sel = true;
                    break;
                }
            }
        } while (already_sel);
        ag->pop_sel_idx[i] = idx;
    }

    for (size_t i = 0; i < ag->sel_size; ++i) {
        ag->pop_sel[i] = ag->pop0[ag->pop_sel_idx[i]];
    }
}


// Função para avançar para a próxima geração
void ag_gen_next(Ag *ag) {
    ag->gen++;

    // elitismo
    memcpy(ag->pop1, ag->pop0, ag->elite_size*sizeof(Ind));

    for (size_t i = ag->elite_size; i < ag->pop_size; i += 2) {
        Ind f1, f2;

        // seleção
        ag_select(ag);
        qsort(ag->pop_sel, ag->sel_size, sizeof(Ind), ind_sort_rev);
        Ind p1 = ag->pop_sel[0];
        Ind p2 = ag->pop_sel[1];

        // cruzamento
        if (randd(0, 1) < ag->tx_cruz) {
            ind_mix(p1, p2, ag->cruz_mask, &f1, &f2);
        } else {
            f1 = p1;
            f2 = p2;            
        }

        // mutação
        f1 = ind_mutate(f1, ag->tx_mut);
        f2 = ind_mutate(f2, ag->tx_mut);

        // reavaliação
        f1.fit = f(num_to_double(f1.x), num_to_double(f1.y));
        f2.fit = f(num_to_double(f2.x), num_to_double(f2.y));

        ag->pop1[i] = f1;
        ag->pop1[i+1] = f2;
    }
    Ind *swap = ag->pop0;
    ag->pop0 = ag->pop1;
    ag->pop1 = swap;
    qsort(ag->pop0, ag->pop_size, sizeof(Ind), ind_sort_rev);
}
//========================================================================


//========================================================================
// Funções para visualização com a biblioteca SDL2
//========================================================================
typedef struct {
    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;
    size_t wt, ht;
    
    uint8_t *keys0, *keys1;
    int numkeys;
} Win;


typedef struct {
    uint8_t r, g, b, a;
} Rgba;


Win win_start(size_t wt, size_t ht) {
    Win win;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(wt, ht, 0, &win.window, &win.renderer);
    SDL_SetRenderDrawColor(win.renderer, 0, 0, 0, 255);
    SDL_RenderClear(win.renderer);
    SDL_RenderPresent(win.renderer);

    win.wt = wt;
    win.ht = ht;

    const uint8_t *keys = SDL_GetKeyboardState(&win.numkeys);
    win.keys1 = calloc(win.numkeys, 1);
    memcpy(win.keys1, keys, win.numkeys);
    win.keys0 = calloc(win.numkeys, 1);

    return win;
}


void win_fill(Win *win, Rgba color, bool blend) {
    SDL_SetRenderDrawBlendMode(win->renderer,
        blend ? SDL_BLENDMODE_BLEND : SDL_BLENDMODE_NONE
    );
    SDL_SetRenderDrawColor(win->renderer, color.r, color.g, color.b, color.a);
    for (size_t y = 0; y < win->ht; ++y) {
        for (size_t x = 0; x < win->wt; ++x) {
            SDL_RenderDrawPoint(win->renderer, x, y);
        }
    }
    SDL_RenderPresent(win->renderer);
}


void win_readkeys(Win *win) {
    int numkeys;
    const uint8_t *keys = SDL_GetKeyboardState(&numkeys);
    win->keys0 = realloc(win->keys0, numkeys);
    size_t bytes = numkeys > win->numkeys ? numkeys - win->numkeys : 0;
    memcpy(win->keys0, win->keys1, win->numkeys);
    memset(&win->keys0[win->numkeys - bytes], 0, bytes);
    memcpy(win->keys1, keys, numkeys);
    win->numkeys = numkeys;
}


bool win_is_held(Win *win, int scancode) {
    return win->keys1[scancode];
}


bool win_was_pressed(Win *win, int scancode) {
    return !win->keys0[scancode] && win->keys1[scancode];
}


void win_pixel(Win *win, size_t x, size_t y, Rgba color, bool blend) {
    SDL_SetRenderDrawBlendMode(win->renderer,
        blend ? SDL_BLENDMODE_BLEND : SDL_BLENDMODE_NONE
    );
    SDL_SetRenderDrawColor(win->renderer, color.r, color.g, color.b, color.a);
    
    // int xoff[4] = {  0,  1,  1,  0 };
    // int yoff[4] = {  0,  0,  1,  1 };

    // for (int i = 0; i < 4; ++i) {
    //     SDL_RenderDrawPoint(win->renderer, x + xoff[i], y + yoff[i]);
    // } 
    SDL_RenderDrawPoint(win->renderer, x, y);
}


void win_render(Win *win) {
    SDL_RenderPresent(win->renderer);
}


void win_end(Win win) {
    free(win.keys0);
    free(win.keys1);
    SDL_DestroyRenderer(win.renderer);
    SDL_DestroyWindow(win.window);
    SDL_Quit();
}


void draw_pop(Win *win, Ag *ag) {
    for (size_t i = 0; i < ag->pop_size; ++i) {
        Ind ind = ag->pop0[i];
        size_t x = (size_t) unit_map(num_to_double(ind.x), -10, 10, 0, win->wt-1);
        size_t y = win->ht - (size_t) unit_map(num_to_double(ind.y), -10, 10, 1, win->ht);
        Rgba color = {
            .r = unit_map(ind.fit, 0, 1.01, 64, 255),
            .g = unit_map(ind.fit, 0, 1.01, 0, 16),
            .b = unit_map(ind.fit, 0, 1.01, 64, 255),
            .a = 255,
        };
        win_pixel(win, x, y, color, false);
    }
    win_render(win);
}


void debug_pop(Win *win, Ag *ag, size_t n) {
    printf("%lu melhores da geração %lu: \n", n, ag->gen);
    size_t j = (n <= ag->pop_size) ? n : ag->pop_size; 
    for (int i = 0; i < j; ++i) {
        Ind ind = ag->pop0[i];
        double xd = num_to_double(ind.x);
        double yd = num_to_double(ind.y);
         
       printf("\tf(%lf, %lf) = %lf\t", xd, yd, ind.fit);
       printf("ponto = (%lu %lu)\n", (size_t) unit_map(xd, -10, 10, 0, win->wt), 600 - (size_t) unit_map(yd, -10, 10, 0, win->ht));
   }
}


int main(int argc, char **argv) {
    size_t pop_size = 200;
    size_t sel_size = 16;
    double elite = 0.01;
    double tx_cruz = 0.80;
    double tx_mut = 0.05;
    uint32_t cruz_mask = (1 << 23) | (1 << 16);

    size_t win_w = 600, win_h = 600;

    size_t debug_amount = 10;

    // loop em argv para parametros de CLI
    int err = 0;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "-help")) {
            printf("ARGS:\n");
            printf("  -pop   SIZE\t: alterar tamanho da população\n");
            printf("  -sel   SIZE\t: alterar tamanho da seleção\n");
            printf("  -w     WIDTH\t: alterar dimensões da janela (quadrada)\n");
            printf("  -elite SIZE\t: alterar tamanho da elite\n");
            printf("  -cruz  RATE\t: alterar taxa de cruzamento\n");
            printf("  -mut   RATE\t: alterar taxa de mutação\n");
            printf("  -d     SIZE\t: printar n melhores indivíduos\n");
            printf("CONTROLES:\n");
            printf("  UP\t: avançar 1 geração\n");
            printf("  RIGHT\t: avançar gerações\n");
            printf("  W\t: avançar 1 geração e limpar tela\n");
            printf("  D\t: avançar gerações e limpar tela\n");
            printf("  BACK\t: limpar tela\n");
            return 0;
        }

        if (!strcmp(argv[i], "-pop")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lu", &pop_size)) {
                err = 1; 
                break;
            }
        }
        if (!strcmp(argv[i], "-sel")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lu", &sel_size)) {
                err = 1;
                break;
            }
        }
        if (!strcmp(argv[i], "-w")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lu", &win_w)) {
                err = 1;
                break;
            }
            win_h = win_w;
        }
        if (!strcmp(argv[i], "-elite")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lf", &elite)) {
                err = 1;
                break;
            }
        }
        if (!strcmp(argv[i], "-cruz")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lf", &tx_cruz)) {
                err = 1;
                break;
            }
        }
        if (!strcmp(argv[i], "-mut")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lf", &tx_mut)) {
                err = 1;
                break;
            }
        }
        if (!strcmp(argv[i], "-d")) {
            i++;
            if (i >= argc || !sscanf(argv[i], "%lu", &debug_amount)) {
                err = 1;
                break;
            }
            if (debug_amount > pop_size) {
                debug_amount = pop_size;
            }
        }
    }
    if (err) {
        printf("Erro ao ler parâmetros de CLI!\n");
        return 1;
    }
    
    
    Win win = win_start(win_w, win_h);
    Ag ag = ag_init(pop_size, sel_size, elite, tx_cruz, tx_mut, cruz_mask);
    srand(time(NULL));

    // renderizando pop inicial
    draw_pop(&win, &ag);
    debug_pop(&win, &ag, debug_amount);


    for (;;) {
        win_readkeys(&win);

        if (SDL_PollEvent(&win.event) && win.event.type == SDL_QUIT || win_is_held(&win, SDL_SCANCODE_ESCAPE)) {
            ag_free(ag);
            win_end(win);
            return 0;
        }

        // avançando para a proxima geração sem limpar tela
        if (win_is_held(&win, SDL_SCANCODE_RIGHT) || win_was_pressed(&win, SDL_SCANCODE_UP)) {
            printf("\nAvançando para geração no. %lu...\n", ag.gen+1);
            ag_gen_next(&ag);
            draw_pop(&win, &ag);
            debug_pop(&win, &ag, debug_amount);
        }

        // limpando e avançando
        if (win_is_held(&win, SDL_SCANCODE_D) || win_was_pressed(&win, SDL_SCANCODE_W)) {
            SDL_SetRenderDrawColor(win.renderer, 0, 0, 0, 0);
            SDL_RenderClear(win.renderer);
            printf("\nAvançando para geração no. %lu...\n", ag.gen+1);
            ag_gen_next(&ag);
            draw_pop(&win, &ag);
            debug_pop(&win, &ag, debug_amount);
        }
        
        // limpando a tela
        if (win_was_pressed(&win, SDL_SCANCODE_BACKSPACE)) {
            SDL_SetRenderDrawColor(win.renderer, 0, 0, 0, 0);
            SDL_RenderClear(win.renderer);
            SDL_RenderPresent(win.renderer);
        }

        SDL_Delay(1000 / 60);
    }
}

